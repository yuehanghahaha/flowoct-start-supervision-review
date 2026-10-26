"""Audit data and time budget, then test two real-size GPU updates and Euler8.

All outputs stay under verification/. This never starts the 100-epoch run.
"""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch

from fm3d.data import VolumeDataset, read_rows, validate_manifest
from fm3d.inference import generate_volume
from fm3d.model import flow_loss, make_model
from fm3d.train import save_checkpoint, seed_all
from fm3d.train_start import (apply_update, file_hash, fixed_probes, layers_for,
                              parse_args, quality, time_schedule, write_json)
from run import ROOT, load_config, training_command


def audit(args, report):
    config = load_config()
    config['layers_dir'] = str(ROOT / config['layers_dir'])
    config['output'] = str(ROOT / config['output'])
    cli = parse_args(training_command(load_config(), args.manifest)[4:])
    assert (cli.epochs, cli.steps_per_epoch, cli.start_fraction) == (100, 200, .5)
    rows = read_rows(args.manifest)
    validate_manifest(rows, config['fov'])
    report.update(manifest=str(args.manifest.resolve()), manifest_sha256=file_hash(args.manifest),
                  split_counts=dict(Counter(r['split'] for r in rows)),
                  config_sha256=file_hash(ROOT / 'experiment.json'))
    start_total = random_total = 0
    for epoch in range(1, config['epochs'] + 1):
        times, mask = time_schedule(config['seed'], epoch, config['steps_per_epoch'], config['start_fraction'])
        uniform, _ = time_schedule(config['seed'], epoch, config['steps_per_epoch'], 0)
        assert int(mask.sum()) == 100 and np.count_nonzero(times == 0) == 100
        np.testing.assert_array_equal(times[~mask], uniform[~mask])
        start_total += int(mask.sum())
        random_total += int((~mask).sum())
    report['budget'] = dict(epochs=config['epochs'], successful_updates=start_total+random_total,
                            start_updates=start_total, random_time_updates=random_total,
                            validations=config['epochs']//config['val_every'],
                            best_snapshots=config['epochs']//config['save_every'])
    annotation_manifest = json.loads((Path(config['layers_dir']) / 'provenance.json').read_text(encoding='utf-8-sig'))
    for annotation in annotation_manifest['cases']:
        assert file_hash(Path(config['layers_dir']) / (annotation['case_id'] + '.mat')) == annotation['sha256']
    training = VolumeDataset(rows, 'train', config['axis_order'], config['intensity_scale'], config['depth'], config['patch'])
    validation = VolumeDataset(rows, 'validation', config['axis_order'], config['intensity_scale'], config['depth'])
    for row in validation.rows:
        layers_for(config, row['case_id'], None)
    probes = fixed_probes(training, validation, config)
    report['data_probes'] = []
    for probe in probes:
        previous = Path('/docsystem/niuke/lvyuehang/fm_start_control_20261022/inputs') / (
            f'{probe["split"]}_{probe["case_id"]}_y064_x064.npz')
        match = None
        if previous.is_file():
            with np.load(previous, allow_pickle=False) as cached:
                match = bool(np.array_equal(probe['source'].numpy(), cached['oct']) and
                             np.array_equal(probe['target'].numpy(), cached['octa']))
            assert match, 'Preprocessing differs from the previous frozen diagnostic input'
        report['data_probes'].append(dict(case_id=probe['case_id'], split=probe['split'],
            shape=list(probe['source'].shape), previous_frozen_input_exact_match=match))
    report['evaluation_annotation_hashes_verified'] = len(annotation_manifest['cases'])
    report['data_and_schedule_checks'] = 'passed'
    write_json(args.output / 'report.json', report)

    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    seed_all(config['seed'])
    device = torch.device(config['device'])
    if device.type != 'cuda':
        raise ValueError('The deployment smoke check must use CUDA')
    model = make_model(config['base_channels'], True, 'voxel').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=config['amp'])
    torch.cuda.reset_peak_memory_stats(device)
    source, target = probes[0]['source'].to(device), probes[0]['target'].to(device)
    update_records = []
    for value in (0., .5):
        for attempt in range(1, 102):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=scaler.is_enabled()):
                loss = flow_loss(model, source, target, torch.tensor([value], device=device))
            skipped, norm = apply_update(loss, model, optimizer, scaler)
            if not skipped:
                break
        else:
            raise FloatingPointError('GPU smoke check exceeded its AMP retry budget')
        update_records.append(dict(time=value, loss=float(loss.detach()), gradient_norm=norm,
                                   attempts=attempt, successful=True))
    prediction = generate_volume(model, probes[0]['source'], config['patch'],
                                 config['ode_steps'], config['overlap'], amp=False)
    assert torch.isfinite(prediction).all() and prediction.shape == probes[0]['target'].shape
    checkpoint = args.output / 'smoke_only.pt'
    save_checkpoint(checkpoint, dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                    scaler=scaler.state_dict(), smoke_only=True))
    restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(restored['model'])
    optimizer.load_state_dict(restored['optimizer'])
    scaler.load_state_dict(restored['scaler'])
    repeated = generate_volume(model, probes[0]['source'], config['patch'],
                               config['ode_steps'], config['overlap'], amp=False)
    torch.testing.assert_close(prediction, repeated, rtol=0, atol=0)
    metrics, _ = quality(prediction, probes[0]['target'], probes[0]['layers'])
    torch.cuda.synchronize(device)
    report.update(state='passed', gpu=torch.cuda.get_device_name(device),
        versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'Pillow', 'scipy', 'scikit-image')},
        parameters=sum(p.numel() for p in model.parameters()), gpu_updates=update_records,
        euler_steps=config['ode_steps'], checkpoint_roundtrip_exact=True,
        smoke_metrics=metrics, peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
        full_training_started=(Path(config['output']) / 'latest.pt').exists(),
        interpretation='Two-update implementation smoke check only; no evidence of convergence or generalization; no test case loaded')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(state='checking', checked_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
    try:
        audit(args, report)
    except Exception as error:
        report.update(state='failed', error_type=type(error).__name__, error=str(error))
        write_json(args.output / 'report.json', report)
        raise
    write_json(args.output / 'report.json', report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

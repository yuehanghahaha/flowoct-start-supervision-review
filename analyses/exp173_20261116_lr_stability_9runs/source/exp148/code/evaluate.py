"""Fixed validation only, with recoverable per-patient 2D evidence."""
import csv
from pathlib import Path
import time

import numpy as np
import torch

import common as c
import method
from metrics import PROTOCOL, quality


@torch.inference_mode()
def validate(model, inputs, rows, cfg, epoch, weights_digest):
    selected = [row for row in rows if row['split'] == 'validation']
    if len(selected) != 10 or len({row['patient_id'] for row in selected}) != 10:
        raise ValueError('Validation requires the fixed 10 patients')
    tag = f'epoch_{epoch:04d}_{weights_digest[:12]}'
    folder = c.ROOT / 'reports/validation' / tag
    (folder / 'views').mkdir(parents=True, exist_ok=True)
    (folder / 'figures').mkdir(exist_ok=True)
    identity = dict(experiment_id=cfg['experiment_id'], split='validation', epoch=epoch,
                    model_tensor_sha256=weights_digest, source_contract=c.source_contract(),
                    runtime_versions=c.versions(), precision_flags=c.precision_flags(model),
                    inference_protocol=cfg['inference_description'], metric_protocol=PROTOCOL)
    identity_path = folder / 'identity.json'
    if identity_path.exists() and c.read(identity_path) != identity:
        raise ValueError('Validation output identity differs')
    c.write(identity_path, identity)
    records = []
    for index, row in enumerate(selected):
        tick = time.perf_counter()
        cid = row['case_id']
        source, target = inputs.pair(row)
        layers = inputs.layers(cid)
        regions = inputs.regions_xy(cid)
        paths = [row['path'], inputs.cache / cid / 'oct.npy', inputs.cache / cid / 'octa.npy']
        paths += [inputs.labels / group / (cid + suffix) for group, suffix in
                  [('GT_Layers', '.mat'), ('GT_Capillary', '.bmp'), ('GT_LargeVessel', '.bmp')]]
        dependencies = {str(Path(path).resolve()): inputs.identify(path) for path in paths}
        record_path = folder / (cid + '.json')
        if record_path.exists():
            record = c.read(record_path)
            if record['input_identities'] != dependencies:
                raise ValueError('Validation result input identity changed for ' + cid)
            if all((folder / name).is_file() and c.sha(folder / name) == digest
                   for name, digest in record['report_files'].items()):
                records.append(record)
                continue

        prediction = method.predict(model, torch.from_numpy(np.array(source, copy=True))[None], cfg)
        pred, gt = prediction[0, 0].numpy(), target[0]
        if pred.shape != (256, 304, 304) or pred.dtype != np.float32:
            raise ValueError('Expected full-resolution float32 OCTA')
        metrics, p, g, bscans = quality(pred, gt, layers, regions, include_reference=False)
        projections = {**{'pred_'+k: v for k, v in p.items()}, **{'gt_'+k: v for k, v in g.items()}}
        views = f'views/{cid}.npz'
        slices = f'views/{cid}_bscans.csv'
        np.savez_compressed(folder / views, **projections)
        with (folder / slices).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['y', 'mae', 'mse', 'psnr', 'ssim'])
            writer.writeheader()
            writer.writerows(bscans)
        evidence = [views, slices]
        if index == 0:
            figure = f'figures/{cid}.png'
            c.case_figure(folder / figure, cid, p, g, pred, gt)
            evidence.append(figure)
        record = dict(case_id=cid, patient_id=row['patient_id'], metrics=metrics, shape_ZYX=list(pred.shape),
                      input_identities=dependencies, report_files={name: c.sha(folder / name) for name in evidence},
                      seconds=time.perf_counter()-tick)
        c.write(record_path, record)
        records.append(record)
        del prediction, pred, gt
        print(f'Validation {index+1}/10 {cid}: 3D SSIM={metrics["volume_ssim"]:.5f}, '
              f'ILM-OPL SSIM={metrics["projection_ilm_opl_ssim"]:.5f}', flush=True)

    result = dict(experiment_id=cfg['experiment_id'], split='validation', epoch=epoch,
                  model_tensor_sha256=weights_digest, **c.patient_summary(records), config=cfg,
                  identity=identity, protocol=PROTOCOL, full_3D_files_saved=False)
    c.evaluation_report(folder, result)
    c.write(folder / 'metrics.json', result)
    return result, folder.relative_to(c.ROOT).as_posix()

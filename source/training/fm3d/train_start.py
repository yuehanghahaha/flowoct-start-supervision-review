"""Voxel 3D FM with explicit start-time learning; 100 epochs, best snapshots every10."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from scipy.io import loadmat
from skimage.metrics import structural_similarity
import torch
from torch.nn import functional as F

from .data import COUNTS, VolumeDataset, read_rows, validate_manifest
from .inference import generate_volume
from .model import flow_loss, make_model
from .projection_report import project
from .train import save_checkpoint, seed_all


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def time_schedule(seed, epoch, steps, start_fraction):
    """Separate RNG: changing time allocation never changes case/crop sampling."""
    if steps < 1 or not 0 <= start_fraction <= 1:
        raise ValueError('steps>=1 and start_fraction in [0,1] required')
    rng = np.random.default_rng(seed + epoch + 100000)
    times = rng.random(steps).astype(np.float32)
    times = np.clip(times, np.nextafter(np.float32(0), np.float32(1)),
                    np.nextafter(np.float32(1), np.float32(0)))
    mask = np.zeros(steps, dtype=bool)
    mask[rng.permutation(steps)[:round(steps * start_fraction)]] = True
    times[mask] = 0.
    return times, mask


def apply_update(loss, model, optimizer, scaler):
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss')
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                         error_if_nonfinite=not scaler.is_enabled())
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = scaler.get_scale() < previous_scale
    if not skipped and (not math.isfinite(float(norm)) or float(norm) <= 0):
        raise FloatingPointError('Nonfinite or zero total gradient')
    return skipped, float(norm)


def map_scores(prediction, target):
    a, b = prediction.astype(np.float64), target.astype(np.float64)
    x, y = a-a.mean(), b-b.mean()
    denominator = float(np.sqrt(np.sum(x*x)*np.sum(y*y)))
    return dict(mae=float(np.abs(a-b).mean()),
                pcc=float(np.sum(x*y))/denominator if denominator > 1e-15 else None,
                ssim=float(structural_similarity(a, b, data_range=1., win_size=7)))


def layers_for(config, case_id, shape):
    if not config['layers_dir']:
        return None
    layers = np.asarray(loadmat(Path(config['layers_dir'])/(case_id+'.mat'))['Layer'], dtype=np.float32)
    if layers.ndim != 3 or layers.shape[0] != 6 or (shape is not None and layers.shape != (6, *shape)):
        raise ValueError(f'{case_id}: expected layer geometry (6,Y,X) matching the volume')
    if not np.isfinite(layers).all() or (layers < 1).any() or (layers > 640).any() or (np.diff(layers, axis=0) < 0).any():
        raise ValueError(f'{case_id}: invalid native640 layer boundaries')
    return layers


def quality(prediction, target, layers=None):
    prediction, target = prediction.detach().cpu(), target.detach().cpu()
    error = prediction.double()-target.double()
    result = dict(mae=float(error.abs().mean()), mse=float(error.square().mean()))
    result['psnr'] = -10*math.log10(max(result['mse'], 1e-12))
    a, b = prediction[0, 0].numpy(), target[0, 0].numpy()
    views = dict(zmax=(a.max(0), b.max(0)))
    if layers is not None:
        pa, empty = project(a, layers)
        pb, _ = project(b, layers)
        views['ilm_opl'] = (pa['ILM_OPL'], pb['ILM_OPL'])
        result['ilm_opl_empty_rays'] = empty['ILM_OPL']
    for name, (a, b) in views.items():
        result.update({name+'_'+key: value for key, value in map_scores(a, b).items()})
    return result, views


def patient_mean(records):
    patients = defaultdict(list)
    for record in records:
        patients[record['patient_id']].append(record['metrics'])
    def average(rows):
        return {key: float(np.mean([r[key] for r in rows if r[key] is not None]))
                if any(r[key] is not None for r in rows) else None for key in rows[0]}
    per_patient = {pid: average(rows) for pid, rows in patients.items()}
    return dict(metrics=average(list(per_patient.values())), per_patient=per_patient,
                patients=len(patients), cases=records)


@torch.inference_mode()
def validate(model, dataset, config):
    records = []
    for index, row in enumerate(dataset.rows):
        sample = dataset[index]
        source, target = sample['oct'][None], sample['octa'][None]
        prediction = generate_volume(model, source, config['patch'], config['ode_steps'],
                                     config['overlap'], amp=False)
        layers = layers_for(config, row['case_id'], target.shape[-2:])
        metrics, _ = quality(prediction, target, layers)
        records.append(dict(case_id=row['case_id'], patient_id=row['patient_id'], metrics=metrics))
        print(f'Validation {index+1}/{len(dataset)} {row["case_id"]}: '
              f'MAE={metrics["mae"]:.6f} zmax SSIM={metrics["zmax_ssim"]:.4f}', flush=True)
    return patient_mean(records)


def fixed_probes(training, validation, config):
    """One predeclared patch per split; never use these patches to select best."""
    probes = []
    for split, dataset in (('train', training), ('validation', validation)):
        row = dataset.rows[0]
        full = VolumeDataset([row], split, config['axis_order'], config['intensity_scale'], config['depth'])[0]
        shape = full['oct'].shape[1:]
        if any(n < p for n, p in zip(shape, config['patch'])):
            raise ValueError(f'{row["case_id"]}: probe patch larger than volume')
        origin = [0, min(64, shape[1]-config['patch'][1]), min(64, shape[2]-config['patch'][2])]
        region = (slice(None),)+tuple(slice(s, s+p) for s, p in zip(origin, config['patch']))
        layers = layers_for(config, row['case_id'], shape[-2:])
        if layers is not None:
            layers = layers[:, region[-2], region[-1]].copy()
        probes.append(dict(case_id=row['case_id'], patient_id=row['patient_id'], split=split,
            origin_ZYX=origin, source=full['oct'][region].contiguous()[None],
            target=full['octa'][region].contiguous()[None], layers=layers))
    return probes


@torch.inference_mode()
def diagnose(model, probes, config, epoch):
    model.eval()
    device = next(model.parameters()).device
    records = []
    folder = Path(config['output'])/'diagnostics'
    folder.mkdir(exist_ok=True)
    for sample in probes:
        source, target = sample['source'].to(device), sample['target'].to(device)
        teacher = []
        for value in (0., .01, .05, .1, .5, .9):
            state = (1-value)*source+value*target
            velocity = model(state, torch.full((1,), value, device=device), source).float()
            teacher.append(dict(time=value, velocity_mse=float(F.mse_loss(velocity, target-source))))
        outputs = {}
        for steps in (1, config['ode_steps']):
            prediction = generate_volume(model, sample['source'], config['patch'], steps,
                                         config['overlap'], amp=False)
            metrics, views = quality(prediction, sample['target'], sample['layers'])
            outputs[str(steps)] = metrics
            if steps == config['ode_steps'] and (epoch % config['save_every'] == 0 or epoch == config['epochs']):
                name = 'ilm_opl' if 'ilm_opl' in views else 'zmax'
                pred, truth = views[name]
                image = np.concatenate((truth, pred), axis=1)
                Image.fromarray(np.rint(image.clip(0, 1)*255).astype(np.uint8)).save(
                    folder/f'{sample["split"]}_{sample["case_id"]}_epoch_{epoch:04d}_{name}_GT_PRED.png')
        records.append(dict(case_id=sample['case_id'], split=sample['split'],
                            origin_ZYX=sample['origin_ZYX'], teacher=teacher, outputs=outputs))
    write_json(folder/f'epoch_{epoch:04d}.json', dict(epoch=epoch, records=records,
        protocol='FP32 probes; t>0 teacher states contain true OCTA; outputs start from OCT only; fixed [0,1]'))
    return records


def snapshot_best(output, payload):
    """Only inference weights for best; latest retains optimizer and best weights for recovery."""
    if payload['best'] is None:
        return
    best_payload = dict(model=payload['best_model'], epoch=payload['best']['epoch'],
        best=payload['best'], config=payload['config'], manifest=payload['manifest'])
    save_checkpoint(output/'best.pt', best_payload)
    epoch = payload['epoch']
    config = payload['config']
    if epoch % config['save_every'] == 0 or epoch == config['epochs']:
        best_file = output/f'best_through_{epoch:04d}.pt'
        shutil.copyfile(output/'best.pt', best_file.with_suffix('.tmp'))
        best_file.with_suffix('.tmp').replace(best_file)
        write_json(output/f'best_through_{epoch:04d}.json', dict(through_epoch=epoch,
            best_epoch=payload['best']['epoch'], metric=payload['best']['metric'],
            value=payload['best']['value'], direction='maximize',
            checkpoint=best_file.name, checkpoint_sha256=file_hash(best_file),
            updates=payload['updates'], val_every=config['val_every']))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--fov', choices=COUNTS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--axis-order', required=True)
    parser.add_argument('--intensity-scale', type=float, required=True)
    parser.add_argument('--depth', type=int, default=256)
    parser.add_argument('--patch', type=int, nargs=3, default=[256, 64, 64])
    parser.add_argument('--base-channels', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps-per-epoch', type=int, default=200, help='Successful optimizer updates')
    parser.add_argument('--start-fraction', type=float, default=.5, help='Exact fraction of updates at t=0 per epoch')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--val-every', type=int, default=5, help='Whole-volume validation frequency')
    parser.add_argument('--save-every', type=int, default=10, help='Cumulative validation-best snapshots')
    parser.add_argument('--ode-steps', type=int, default=8)
    parser.add_argument('--overlap', type=float, default=.5)
    parser.add_argument('--layers-dir', type=Path, help='Optional native640 Layer MAT files, evaluation only')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', default='cuda', help='Use --device cpu explicitly for small implementation checks')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if min(args.epochs, args.steps_per_epoch, args.val_every, args.save_every, args.ode_steps) < 1:
        raise ValueError('Epoch/update/frequency/ODE counts must be positive')
    if args.depth < 8 or min(args.patch) < 8 or args.patch[0] != args.depth:
        raise ValueError('patch>=8 and patch Z must span the resampled depth')
    if not 0 <= args.start_fraction < 1:
        raise ValueError('start-fraction must be in [0,1); retain random-time FM updates')
    starts = round(args.steps_per_epoch*args.start_fraction)
    if args.start_fraction and not 0 < starts < args.steps_per_epoch:
        raise ValueError('Epoch must contain at least one start and one random-time update')
    if args.save_every % args.val_every:
        raise ValueError('save-every must be divisible by val-every')
    if not 0 <= args.overlap < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError('Invalid overlap or learning rate')
    if sorted(args.axis_order.upper()) != ['X', 'Y', 'Z'] or not math.isfinite(args.intensity_scale) or args.intensity_scale <= 0:
        raise ValueError('Explicit axes and positive finite intensity scale required')
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = read_rows(args.manifest)
    validate_manifest(rows, args.fov)
    config = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(axis_order=args.axis_order.upper(), representation='voxel', condition=True, batch_size=1,
        best_metric='ilm_opl_ssim' if args.layers_dir else 'zmax_ssim',
        manifest_sha256=file_hash(args.manifest), lambda_projection=0.,
        source_sha256={name: file_hash(Path(__file__).parent/name) for name in
                       ('train_start.py', 'train.py', 'model.py', 'data.py', 'inference.py',
                        'wavelets.py', 'projection_report.py')})
    training = VolumeDataset(rows, 'train', config['axis_order'], args.intensity_scale, args.depth, args.patch)
    validation = VolumeDataset(rows, 'validation', config['axis_order'], args.intensity_scale, args.depth)
    if args.layers_dir:
        annotation_ids = [training.rows[0]['case_id']]+[r['case_id'] for r in validation.rows]
        config['evaluation_layer_sha256'] = {cid: file_hash(args.layers_dir/(cid+'.mat')) for cid in annotation_ids}
    seed_all(args.seed)
    device = torch.device(args.device)
    model = make_model(args.base_channels, True, 'voxel').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024, enabled=args.amp and device.type == 'cuda')
    first_epoch, updates, attempts, best, best_model = 1, 0, 0, None, None
    latest = args.output/'latest.pt'
    if args.resume:
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        for key, value in config.items():
            if key not in ('epochs', 'resume', 'device', 'output') and saved['config'].get(key) != value:
                raise ValueError(f'Resume protocol changed: {key}')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        first_epoch, updates, attempts = saved['epoch']+1, saved['updates'], saved['attempts']
        best, best_model = saved['best'], saved['best_model']
        snapshot_best(args.output, saved)
        for name in ('steps.jsonl', 'history.jsonl'):
            path = args.output/name
            content = path.read_text(encoding='utf-8') if path.exists() else ''
            records = [json.loads(line) for line in content.splitlines(keepends=True)
                       if line.endswith('\n') and line.strip()]
            records = [r for r in records if r['epoch'] < first_epoch]
            if name == 'history.jsonl' and not any(r['epoch'] == saved['epoch'] for r in records):
                records.append(saved['epoch_record'])
            path.write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in records), encoding='utf-8')
        del saved
    elif args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output is not empty; use a new directory or --resume')
    if first_epoch > args.epochs:
        raise ValueError('Training budget already completed')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'config.json', config)
    write_json(args.output/'manifest_snapshot.json', rows)
    write_json(args.output/'environment.json', dict(torch=torch.__version__, numpy=np.__version__,
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        parameters=sum(p.numel() for p in model.parameters()),
        protocol='No wavelets or auxiliary losses; fixed LR; eval FP32; one forward/backward per successful update'))
    probes = fixed_probes(training, validation, config)
    write_json(args.output/'probe_selection.json', [{k: sample[k] for k in
               ('case_id', 'patient_id', 'split', 'origin_ZYX')} for sample in probes])
    # Preflight annotation geometry before committing any training.
    if args.layers_dir:
        for row in validation.rows:
            layers_for(config, row['case_id'], None)
    write_json(args.output/'status.json', dict(state='running', epoch=first_epoch-1,
        epochs=args.epochs, updates=updates, best=best))
    print(f'3DFM+start epochs={first_epoch}..{args.epochs} successful_updates/epoch={args.steps_per_epoch} '
          f'start_updates/epoch={starts} best_metric={config["best_metric"]}', flush=True)
    for epoch in range(first_epoch, args.epochs+1):
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        seed_all(args.seed+epoch)
        rng = np.random.default_rng(args.seed+epoch)
        indices = rng.integers(len(training), size=args.steps_per_epoch)
        times, start_mask = time_schedule(args.seed, epoch, args.steps_per_epoch, args.start_fraction)
        model.train()
        losses, grouped = [], {'start': [], 'random': []}
        skipped, tick = 0, time.perf_counter()
        with (args.output/'steps.jsonl').open('a', encoding='utf-8', buffering=1) as stream:
            for index, (case_index, value) in enumerate(zip(indices, times)):
                sample = training[int(case_index)]
                source, target = sample['oct'][None].to(device), sample['octa'][None].to(device)
                t = torch.tensor([float(value)], device=device)
                while True:
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                        loss = flow_loss(model, source, target, t)
                    overflow, grad_norm = apply_update(loss, model, optimizer, scaler)
                    attempts += 1
                    if not overflow:
                        break
                    skipped += 1
                    if skipped > 100:
                        raise FloatingPointError('Repeated AMP overflows')
                updates += 1
                value_loss = float(loss.detach())
                group = 'start' if start_mask[index] else 'random'
                losses.append(value_loss)
                grouped[group].append(value_loss)
                stream.write(json.dumps(dict(epoch=epoch, step=index+1, update=updates, attempt=attempts,
                    case_id=sample['case_id'], time=float(value), group=group, loss=value_loss,
                    gradient_norm=grad_norm), allow_nan=False)+'\n')
                if (index+1) % 20 == 0 or index == 0:
                    print(f'Epoch {epoch}/{args.epochs} update {index+1}/{args.steps_per_epoch} '
                          f'{group} loss={value_loss:.6f}', flush=True)
        diagnostics = diagnose(model, probes, config, epoch)
        report, improved = None, False
        if epoch % args.val_every == 0 or epoch == args.epochs:
            report = validate(model, validation, config)
            write_json(args.output/f'validation_{epoch:04d}.json', report)
            value = report['metrics'][config['best_metric']]
            if value is None or not math.isfinite(value):
                raise FloatingPointError('Invalid validation selection metric')
            improved = best is None or value > best['value']
            if improved:
                best = dict(epoch=epoch, metric=config['best_metric'], value=value)
                best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        record = dict(epoch=epoch, updates=updates, attempts=attempts, loss=float(np.mean(losses)),
            start_loss=float(np.mean(grouped['start'])) if grouped['start'] else None,
            random_time_loss=float(np.mean(grouped['random'])) if grouped['random'] else None,
            start_updates=len(grouped['start']), random_time_updates=len(grouped['random']),
            amp_skipped=skipped, amp_scale=scaler.get_scale(), seconds=time.perf_counter()-tick,
            peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30 if device.type == 'cuda' else None,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30 if device.type == 'cuda' else None,
            best=best, validation=report['metrics'] if report else None,
            diagnostics=diagnostics)
        payload = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
            config=config, manifest=rows, epoch=epoch, updates=updates, attempts=attempts,
            best=best, best_model=best_model, epoch_record=record)
        save_checkpoint(latest, payload)
        if improved or epoch % args.save_every == 0 or epoch == args.epochs:
            snapshot_best(args.output, payload)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_checkpoint(args.output/f'epoch_{epoch:04d}.pt', payload)
        with (args.output/'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        write_json(args.output/'status.json', dict(state='complete' if epoch == args.epochs else 'running',
            epoch=epoch, epochs=args.epochs, updates=updates, best=best))
        print(f'Epoch {epoch}/{args.epochs} complete, updates={updates}, best={best}', flush=True)


if __name__ == '__main__':
    main()

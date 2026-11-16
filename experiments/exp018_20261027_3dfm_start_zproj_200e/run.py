"""Formal FP32 3DFM training, complete-volume evaluation and frozen final test."""
import os
os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'

import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
import hashlib
import html
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image
from scipy.io import loadmat
from skimage.metrics import structural_similarity
import torch
from torch.nn import functional as F

from fm3d.data import load_pair, read_rows
from fm3d.model import make_model
from fm3d.inference import generate_volume
from fm3d.projection_report import project

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/train'
CKPT = ROOT / 'checkpoints'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(path, payload):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(payload, tmp); tmp.replace(path)


def source_contract():
    paths = [ROOT / 'run.py', ROOT / 'experiment.json', ROOT / 'membership.json']
    paths += sorted((ROOT / 'fm3d').glob('*.py'))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def model_digest(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        a = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode()); h.update(a.numpy().tobytes())
    return h.hexdigest()


def versions():
    return {name: importlib.metadata.version(name) for name in
            ['torch', 'numpy', 'scipy', 'scikit-image', 'Pillow', 'matplotlib']}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure(config):
    if config['amp'] or config['gradient_scaling'] or config['tf32']:
        raise ValueError('This frozen experiment requires FP32, no AMP/scaler/TF32')
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(config['threads'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; formal training does not fall back to CPU')


@contextmanager
def training_lock():
    if os.name != 'posix':
        raise RuntimeError('Use the Linux A100 environment for formal training')
    import fcntl
    path = ROOT / 'runs/training.lock'
    with path.open('a+', encoding='utf-8') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This experiment already has an active training process')
        stream.seek(0); stream.truncate(); stream.write(str(os.getpid())); stream.flush()
        yield


def binding(args):
    path = ROOT / 'deployment.json'
    if args.data_root or args.manifest or args.labels_root or not path.exists():
        previous = {}
        for pattern in ['exp014_*/deployment.json', 'exp013_*/deployment.json', 'exp011_*/deployment.json', 'exp004_*/deployment.json']:
            for candidate in sorted(ROOT.parent.glob(pattern)):
                old = read(candidate)
                if old.get('manifest') and Path(old['manifest']).is_file():
                    previous = old; break
            if previous: break
        data = Path(args.data_root or previous.get('data_dir') or '/workspace/flowoct_work/data/OCTA500_v1').resolve()
        manifest = args.manifest or (data/'split_3M_z256.csv' if args.data_root else previous.get('manifest', data/'split_3M_z256.csv'))
        value = dict(data_root=str(data), manifest=str(Path(manifest).resolve()),
                     labels_root=str(Path(args.labels_root or data / 'labels_native').resolve()))
        if path.exists() and read(path) != value and ((CKPT/'latest.pt').exists() or (CKPT/'initial.pt').exists()):
            raise ValueError('A started run cannot silently change its data binding')
        write(path, value)
    value = read(path)
    rows = read_rows(value['manifest'])
    expected = read(ROOT / 'membership.json')
    actual = [{k: row[k] for k in ['case_id', 'patient_id', 'split']} for row in rows]
    if len(actual) != len(expected) or {r['case_id']: r for r in actual} != {r['case_id']: r for r in expected}:
        raise ValueError('Manifest differs from the fixed 140/10/50 patient membership')
    by_id = {r['case_id']: r for r in rows}
    rows = [by_id[r['case_id']] for r in expected]
    counts = {split: sum(r['split'] == split for r in rows) for split in ['train', 'validation', 'test']}
    if counts != {'train': 140, 'validation': 10, 'test': 50}:
        raise ValueError('Invalid fixed split')
    patients = {}
    paths, groups = set(), {}
    for row in rows:
        if row['patient_id'] in patients and patients[row['patient_id']] != row['split']:
            raise ValueError('Patient crosses splits')
        patients[row['patient_id']] = row['split']
        if row['path'] in paths: raise ValueError('Duplicate image path in fixed manifest')
        paths.add(row['path'])
        group = row.get('leakage_group', '').strip()
        if group:
            if group in groups and groups[group] != row['split']: raise ValueError('Leakage group crosses splits')
            groups[group] = row['split']
    # Required scientific inputs, not a short training or validation-model gate.
    for row in rows:
        if row['split'] == 'test': continue
        if not Path(row['path']).is_file(): raise FileNotFoundError(row['path'])
        names = ['GT_Layers'] if row['split'] == 'validation' or read(ROOT/'experiment.json')['projection_mode'] != 'none' else []
        if row['split'] == 'validation' or read(ROOT/'experiment.json')['projection_mode'] == 'regional':
            names += ['GT_Capillary', 'GT_LargeVessel']
        for name in names:
            suffix = '.mat' if name == 'GT_Layers' else '.bmp'
            required = Path(value['labels_root'])/name/(row['case_id']+suffix)
            if not required.is_file(): raise FileNotFoundError(f'Missing required annotation: {required}; use --labels-root')
    return value, rows


class Inputs:
    def __init__(self, deployment, config):
        self.labels = Path(deployment['labels_root'])
        self.config = config
        self.identity_path = ROOT / 'verification/input_identity.json'
        self.identities = read(self.identity_path) if self.identity_path.exists() else {}
        self.seen = set()

    def identify(self, path):
        path = Path(path); name = str(path.resolve())
        if name not in self.seen:
            digest = sha(path)
            if name in self.identities and self.identities[name]['sha256'] != digest:
                raise ValueError(f'Previously used input bytes changed: {path}')
            self.identities[name] = dict(sha256=digest, bytes=path.stat().st_size)
            self.seen.add(name)
            self.flush()

    def flush(self):
        write(self.identity_path, self.identities)

    def pair(self, row):
        self.identify(row['path'])
        source, target = load_pair(row['path'], 'ZYX', 255., self.config['depth'])
        if source.shape != target.shape or tuple(source.shape) != (1, 256, 304, 304):
            raise ValueError(f'{row["case_id"]}: expected full preprocessed [1,256,304,304]')
        return source, target

    @lru_cache(maxsize=None)
    def layers(self, cid):
        path = self.labels / 'GT_Layers' / (cid + '.mat'); self.identify(path)
        a = np.asarray(loadmat(path)['Layer'], dtype=np.float32)
        if a.shape != (6, 304, 304) or not np.isfinite(a).all():
            raise ValueError(f'{cid}: invalid layer geometry')
        if a.min() < 1 or a.max() > 640 or (np.diff(a, axis=0) < 0).any():
            raise ValueError(f'{cid}: invalid native640 MATLAB-one-based layer positions')
        return a

    @lru_cache(maxsize=None)
    def regions_xy(self, cid):
        arrays = []
        for name in ['GT_Capillary', 'GT_LargeVessel']:
            path = self.labels / name / (cid + '.bmp'); self.identify(path)
            with Image.open(path) as im:
                a = np.asarray(im).copy()
            if a.shape != (304, 304) or not set(np.unique(a)).issubset({0, 255}):
                raise ValueError(f'{cid}/{name}: expected native XY binary 0/255 BMP')
            arrays.append(a > 0)
        vessel, large = arrays
        return dict(nonlarge=vessel & ~large, large=large, background=~(vessel | large))

    def training_patch(self, row):
        source, target = self.pair(row)
        # Match the parent's Python-random crop order, independent of time/loss RNG.
        origin = tuple(random.randint(0, n-p) for n, p in zip(source.shape[1:], self.config['patch']))
        slices = tuple(slice(s, s+p) for s, p in zip(origin, self.config['patch']))
        region = (slice(None),) + slices
        return source[region].contiguous()[None], target[region].contiguous()[None], origin, slices


def time_schedule(config, epoch):
    rng = np.random.default_rng(config['seed'] + epoch + 100000)
    times = rng.random(config['steps_per_epoch']).astype(np.float32)
    times = np.clip(times, np.nextafter(np.float32(0), np.float32(1)),
                    np.nextafter(np.float32(1), np.float32(0)))
    mask = np.zeros(config['steps_per_epoch'], dtype=bool)
    mask[rng.permutation(len(mask))[:round(len(mask)*config['start_fraction'])]] = True
    times[mask] = 0.
    return times, mask


def projection_loss(endpoint, target, layers_yx, region_yx=None):
    # Invalid depths must be -inf, not zero: endpoint remains signed/unclamped.
    z = endpoint.shape[2]
    centers = ((torch.arange(z, device=endpoint.device, dtype=torch.float32)+.5)*640/z-.5)
    layers = torch.as_tensor(layers_yx.copy(), device=endpoint.device, dtype=torch.float32)
    mask = (centers[:, None, None] >= layers[0]-1) & (centers[:, None, None] <= layers[2]-1)
    valid = mask.any(0)
    if not valid.any():
        raise ValueError('Training crop has no valid ILM-OPL depth samples')
    p = endpoint[0, 0].masked_fill(~mask, -torch.inf).max(0).values
    g = target[0, 0].masked_fill(~mask, -torch.inf).max(0).values
    error = (p[valid]-g[valid]).abs()
    global_loss = error.mean()
    nonlarge_loss = endpoint.new_zeros(())
    count = 0
    if region_yx is not None:
        selected = torch.as_tensor(region_yx.copy(), device=endpoint.device, dtype=torch.bool) & valid
        count = int(selected.sum())
        if count:
            nonlarge_loss = (p[selected]-g[selected]).abs().mean()
    return global_loss, nonlarge_loss, count, int(valid.sum())


def scores(pred, gt, data_range=1., with_pcc=False):
    abs_sum = sq_sum = 0.
    for first in range(0, gt.shape[0], 16):
        error = pred[first:first+16].astype(np.float64)-gt[first:first+16].astype(np.float64)
        abs_sum += float(np.abs(error).sum()); sq_sum += float(np.square(error).sum())
    mse = sq_sum / gt.size
    result = dict(mae=abs_sum/gt.size, mse=mse,
                  psnr=-10*math.log10(max(mse, 1e-12)) + 20*math.log10(data_range),
                  ssim=float(structural_similarity(gt, pred, data_range=data_range, win_size=7)))
    if with_pcc:
        a = pred.astype(np.float64)-pred.mean(dtype=np.float64)
        b = gt.astype(np.float64)-gt.mean(dtype=np.float64)
        den = np.sqrt(np.square(a).sum()*np.square(b).sum())
        result['pcc'] = float((a*b).sum()/den) if den > 1e-15 else None
    return result


def quality(pred, gt, layers, regions):
    metrics = {'volume_'+key: value for key, value in scores(pred, gt).items()}
    data_range = float(gt.max()-gt.min())
    metrics['reference_volume_data_range'] = data_range
    if data_range > 0:
        metrics['reference_volume_psnr'] = metrics['volume_psnr'] + 20*math.log10(data_range)
        metrics['reference_volume_ssim'] = metrics['volume_ssim'] if data_range == 1. else float(
            structural_similarity(gt, pred, data_range=data_range, win_size=7))
    else:
        metrics.update(reference_volume_psnr=None, reference_volume_ssim=None)
    p, empty = project(pred, layers); g, _ = project(gt, layers)
    p['ZMAX'], g['ZMAX'] = pred.max(0).T, gt.max(0).T
    for name in p:
        metrics.update({'projection_'+name.lower()+'_'+key: value
                        for key, value in scores(p[name], g[name], with_pcc=True).items()})
    metrics['ilm_opl_empty_rays'] = empty['ILM_OPL']
    err = np.abs(p['ILM_OPL'].astype(np.float64)-g['ILM_OPL'])
    for name, mask in regions.items():
        metrics['region_'+name+'_pixels'] = int(mask.sum())
        metrics['region_'+name+'_mae'] = float(err[mask].mean()) if mask.any() else None
        metrics['region_'+name+'_bias'] = float((p['ILM_OPL']-g['ILM_OPL'])[mask].mean(dtype=np.float64)) if mask.any() else None
    return metrics, p, g


def patient_summary(records):
    patients = defaultdict(list)
    for row in records:
        patients[row['patient_id']].append(row['metrics'])
    keys = list(records[0]['metrics'])
    def average(items):
        return {key: float(np.mean([r[key] for r in items if r[key] is not None]))
                if any(r[key] is not None for r in items) else None for key in keys}
    individual = {pid: average(rows) for pid, rows in patients.items()}
    return dict(patients=len(patients), metrics=average(list(individual.values())),
                per_patient=individual, cases=records)


def case_figure(path, cid, p, g, pred, gt):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(5, 3, figsize=(11, 16), constrained_layout=True)
    pairs = [(g['ILM_OPL'], p['ILM_OPL'], 'ILM-OPL MIP', .5, .2),
             (g['OPL_BM'], p['OPL_BM'], 'OPL-BM MIP', .5, .2),
             (g['FULL'], p['FULL'], 'Full-Z mean', .08, .04),
             (g['ZMAX'], p['ZMAX'], 'Full-Z maximum', .5, .2),
             (gt[:, 152, :], pred[:, 152, :], 'B-scan Y=152', .35, .15)]
    for row, (truth, estimate, title, limit, error_limit) in enumerate(pairs):
        for col, a in enumerate([truth, estimate, np.abs(truth-estimate)]):
            image = axes[row, col].imshow(a, cmap='gray' if col < 2 else 'magma',
                                          vmin=0, vmax=limit if col < 2 else error_limit,
                                          interpolation='nearest')
            axes[row, col].set_title(title+' / '+['GT', 'Prediction', 'Abs error'][col], fontsize=9)
            axes[row, col].axis('off')
            if col == 2: fig.colorbar(image, ax=axes[row, col], shrink=.7)
    fig.suptitle(f'{cid}: raw float metrics, fixed GT/pred display windows')
    fig.savefig(path, dpi=150); plt.close(fig)


def evaluation_report(folder, result):
    import csv
    rows = result['cases']; keys = list(result['metrics'])
    with (folder / 'per_case_metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'patient_id']+keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(case_id=row['case_id'], patient_id=row['patient_id'], **row['metrics']))
    standards = ['volume_mae', 'volume_psnr', 'volume_ssim', 'reference_volume_psnr', 'reference_volume_ssim']
    table = ''.join(f'<tr><td>{html.escape(k)}</td><td>{result["metrics"][k]}</td></tr>' for k in standards)
    cases = ''.join(f'<h2>{r["case_id"]}</h2><img loading="lazy" src="figures/{r["case_id"]}.png">' for r in rows)
    doc = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>3DFM evaluation</title>'
           '<style>body{font:16px/1.7 sans-serif;max-width:1150px;margin:35px auto;padding:20px;color:#203342}'
           'img{max-width:100%}td{border:1px solid #ccd7df;padding:8px}table{border-collapse:collapse}</style>'
           f'<h1>{html.escape(result["experiment_id"])} / {result["split"]}</h1>'
           f'<p>Epoch {result["epoch"]}; patients={result["patients"]}; FP32 Euler8; full 256x304x304.</p>'
           '<p>Volume MAE/PSNR/3D SSIM: fixed range=1. Reference columns: per-patient GT max-min. '
           'Both SSIM windows=7. Equal-patient aggregation. Projection/region values are additional diagnostics.</p>'
           '<p><a href="per_case_metrics.csv">All case metrics</a> | <a href="metrics.json">Complete definitions/results</a></p>'
           '<table>'+table+'</table>'+cases+'</html>')
    (folder / 'index.html').write_text(doc, encoding='utf-8')


@torch.inference_mode()
def evaluate(model, inputs, rows, config, split, epoch, weights_digest):
    selected = [r for r in rows if r['split'] == split]
    expected = 10 if split == 'validation' else 50
    if len(selected) != expected: raise ValueError('Invalid evaluation patient count')
    identity = dict(experiment_id=config['experiment_id'], split=split, epoch=epoch,
                    model_tensor_sha256=weights_digest, source_contract=source_contract(), runtime_versions=versions())
    tag = f'epoch_{epoch:04d}_{weights_digest[:12]}'
    folder = ROOT / 'reports' / split / tag
    output = ROOT / 'predictions' / split / tag
    folder.mkdir(parents=True, exist_ok=True); output.mkdir(parents=True, exist_ok=True)
    (folder / 'figures').mkdir(exist_ok=True); (folder / 'views').mkdir(exist_ok=True)
    if (folder / 'identity.json').exists() and read(folder / 'identity.json') != identity:
        raise ValueError('Evaluation identity mismatch; refusing to reuse output')
    write(folder / 'identity.json', identity)
    records = []
    for i, row in enumerate(selected):
        cid = row['case_id']; record_path = folder / (cid+'.json'); prediction_path = output / (cid+'.npz')
        inputs.identify(row['path']); layers = inputs.layers(cid); regions = inputs.regions_xy(cid)
        input_digest = inputs.identities[str(Path(row['path']).resolve())]['sha256']
        if record_path.exists():
            record = read(record_path)
            if record['input_sha256'] != input_digest or sha(prediction_path) != record['prediction_sha256']:
                raise ValueError('Saved result/source checksum mismatch')
            records.append(record); continue
        source, target = inputs.pair(row)
        prediction = generate_volume(model, source[None], config['patch'], steps=8, overlap=.5, amp=False)
        pred, gt = prediction[0, 0].numpy(), target[0].numpy()
        if pred.dtype != np.float32 or not np.isfinite(pred).all():
            raise FloatingPointError('Invalid full-resolution FP32 prediction')
        temporary = prediction_path.with_name(cid+'.tmp.npz')
        np.savez_compressed(temporary, prediction=pred, axis_order=np.array('ZYX'),
                            case_id=np.array(cid), epoch=np.int32(epoch))
        temporary.replace(prediction_path)
        metrics, p, g = quality(pred, gt, layers, regions)
        np.savez_compressed(folder/'views'/f'{cid}.npz',
                            **{'pred_'+k: v for k, v in p.items()}, **{'gt_'+k: v for k, v in g.items()})
        case_figure(folder/'figures'/f'{cid}.png', cid, p, g, pred, gt)
        record = dict(case_id=cid, patient_id=row['patient_id'], metrics=metrics,
                      input_sha256=input_digest, prediction_sha256=sha(prediction_path),
                      full_prediction=str(prediction_path.relative_to(ROOT)), shape_ZYX=list(pred.shape),
                      model_tensor_sha256=weights_digest)
        write(record_path, record); records.append(record)
        inputs.flush()
        print(f'{split} {i+1}/{len(selected)} {cid}: volume MAE={metrics["volume_mae"]:.6f}, '
              f'PSNR={metrics["volume_psnr"]:.3f}, 3D SSIM={metrics["volume_ssim"]:.4f}', flush=True)
    result = dict(experiment_id=config['experiment_id'], split=split, epoch=epoch,
                  model_tensor_sha256=weights_digest, **patient_summary(records),
                  protocol=dict(normalization='fixed /255 after parent Z256 preprocessing; no per-case stretching',
                    full_prediction='float32 ZYX 256x304x304; all cases saved',
                    standard='full-volume MAE/MSE/PSNR/3D SSIM range=1; SSIM win7 uniform sample covariance',
                    reference='additional PSNR/3D SSIM with per-patient GT max-min, official FlowOCT metric range convention',
                    diagnostics='ILM-OPL/OPL-BM Z maxima, full-Z mean/max, PCC, labeled-region errors',
                    aggregation='per-volume score -> within-patient average -> equal patient mean'))
    write(folder/'metrics.json', result); evaluation_report(folder, result)
    write(ROOT/'reports'/split/'latest_report.json', dict(folder=str(folder.relative_to(ROOT)), epoch=epoch))
    return result, str(folder.relative_to(ROOT))


def restore_logs(updates, epoch):
    for name, field, limit in [('steps.jsonl', 'update', updates), ('history.jsonl', 'epoch', epoch)]:
        path = RUN / name
        if not path.exists(): continue
        lines = path.read_text(encoding='utf-8').splitlines()
        keep, interrupted = [], []
        for line in lines:
            try: valid = json.loads(line)[field] <= limit
            except (ValueError, KeyError): valid = False
            (keep if valid else interrupted).append(line)
        if interrupted:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            (ROOT/'logs'/f'interrupted_{stamp}_{name}').write_text('\n'.join(interrupted)+'\n', encoding='utf-8')
            temp = path.with_suffix('.tmp'); temp.write_text('\n'.join(keep)+'\n' if keep else '', encoding='utf-8'); temp.replace(path)


def train(config, deployment, rows, resume):
    contract = source_contract(); configure(config); seed_all(config['seed'])
    device = torch.device(config['device'])
    inputs = Inputs(deployment, config)
    training = [r for r in rows if r['split'] == 'train']
    model = make_model(16, True, 'voxel').float().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.)
    epoch_done, updates, best, reference_best = 0, 0, None, None
    latest = CKPT / 'latest.pt'
    if resume:
        recovery_path = latest if latest.exists() else CKPT/'initial.pt'
        saved = torch.load(recovery_path, map_location='cpu', weights_only=False)
        if saved['source_contract'] != contract or saved['config'] != config or saved['deployment'] != deployment:
            raise ValueError('Resume source, scientific configuration or data binding changed')
        if saved['runtime_versions'] != versions(): raise ValueError('Resume dependency versions changed')
        if saved['manifest_sha256'] != sha(deployment['manifest']): raise ValueError('Resume data manifest bytes changed')
        model.load_state_dict(saved['model'])
        if saved['epoch'] > 0: optimizer.load_state_dict(saved['optimizer'])
        # Move Adam states back to the model device after loading on CPU.
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value) and key != 'step': state[key] = value.to(device)
        epoch_done, updates = saved['epoch'], saved['updates']
        best, reference_best = saved['best'], saved['reference_best']
        restore_logs(updates, epoch_done)
        del saved
    elif latest.exists() or (CKPT/'initial.pt').exists() or (RUN/'steps.jsonl').exists():
        raise FileExistsError('Started run exists; use resume, never overwrite')
    if epoch_done >= config['epochs']:
        print('Frozen training budget already complete; no new training started.', flush=True); return
    RUN.mkdir(parents=True, exist_ok=True)
    write(RUN/'status.json', dict(state='initializing', completed_epoch=epoch_done,
        epoch=epoch_done, epochs=config['epochs'], updates=updates, best=best, pid=os.getpid()))
    write(RUN/'manifest_snapshot.json', rows)
    initial_digest = model_digest(model)
    if not resume:
        save_checkpoint(CKPT/'initial.pt', dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            epoch=0, updates=0, best=None, reference_best=None, model_tensor_sha256=initial_digest,
            config=config, source_contract=contract, deployment=deployment,
            manifest_sha256=sha(deployment['manifest']), runtime_versions=versions()))
    environment = dict(torch=torch.__version__, numpy=np.__version__,
        versions=versions(), cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(device), parameters=sum(p.numel() for p in model.parameters()),
        precision='FP32; no autocast or GradScaler; both TF32 switches off',
        source_contract=contract, initial_or_resume_tensor_sha256=initial_digest,
        cpu_threads=config['threads'], cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        started_at=datetime.now().astimezone().isoformat(), deployment=deployment,
        filesystem_free_bytes=shutil.disk_usage(ROOT).free)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    write(ROOT/'verification'/f'environment_{stamp}_{os.getpid()}.json', environment)
    write(ROOT/'verification/run_environment.json', environment)
    for epoch in range(epoch_done+1, config['epochs']+1):
        seed_all(config['seed']+epoch)
        indices = np.random.default_rng(config['seed']+epoch).integers(len(training), size=config['steps_per_epoch'])
        times, mask = time_schedule(config, epoch)
        model.train(); torch.cuda.reset_peak_memory_stats(device); tick = time.perf_counter()
        total, fm_values, global_values, fine_values = [], [], [], []
        write(RUN/'status.json', dict(state='training', epoch=epoch, completed_epoch=epoch-1,
                                    epochs=config['epochs'], updates=updates, best=best, pid=os.getpid()))
        with (RUN/'steps.jsonl').open('a', encoding='utf-8', buffering=1) as stream:
            for step, (index, t_value) in enumerate(zip(indices, times), 1):
                row = training[int(index)]
                source, target, origin, slices = inputs.training_patch(row)
                source, target = source.to(device), target.to(device)
                t = torch.tensor([float(t_value)], device=device, dtype=torch.float32)
                state = (1-t[:, None, None, None, None])*source+t[:, None, None, None, None]*target
                optimizer.zero_grad(set_to_none=True)
                velocity = model(state, t, source)
                if velocity.dtype != torch.float32: raise TypeError('Non-FP32 forward is forbidden')
                fm = F.mse_loss(velocity, target-source)
                global_loss, fine_loss = source.new_zeros(()), source.new_zeros(())
                n_fine = n_valid = 0
                if mask[step-1] and config['projection_mode'] != 'none':
                    layers = inputs.layers(row['case_id'])[:, slices[1], slices[2]]
                    selected = None
                    if config['projection_mode'] == 'regional':
                        # BMPs are XY; training tensors/layer crops are YX.
                        selected = inputs.regions_xy(row['case_id'])['nonlarge'].T[slices[1], slices[2]]
                    global_loss, fine_loss, n_fine, n_valid = projection_loss(source+velocity, target, layers, selected)
                loss = fm + config['lambda_projection']*(global_loss+config['regional_alpha']*fine_loss)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss; no skipped/reduced-precision updates')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                if not math.isfinite(float(norm)) or float(norm) <= 0: raise FloatingPointError('Invalid total gradient')
                optimizer.step(); updates += 1
                values = [float(x.detach()) for x in [loss, fm, global_loss, fine_loss]]
                total.append(values[0]); fm_values.append(values[1])
                if mask[step-1] and config['projection_mode'] != 'none':
                    global_values.append(values[2]); fine_values.append(values[3])
                stream.write(json.dumps(dict(epoch=epoch, step=step, update=updates, case_id=row['case_id'],
                    origin_ZYX=origin, time=float(t_value), exact_start=bool(mask[step-1]),
                    loss=values[0], fm_mse=values[1], global_projection_l1=values[2], nonlarge_projection_l1=values[3],
                    nonlarge_pixels=n_fine, valid_slab_rays=n_valid, gradient_norm=float(norm)), allow_nan=False)+'\n')
                if step == 1 or step % 20 == 0:
                    write(RUN/'status.json', dict(state='training', epoch=epoch, completed_epoch=epoch-1,
                        epochs=config['epochs'], step=step, updates=updates, best=best, pid=os.getpid()))
                    print(f'Epoch {epoch}/{config["epochs"]} step {step}/200: FM={values[1]:.6f} '
                          f'Proj={values[2]:.6f} Nonlarge={values[3]:.6f}', flush=True)
        digest = model_digest(model)
        # This checkpoint preserves the trained epoch even if validation is interrupted.
        trained = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                       updates=updates, best=best, reference_best=reference_best,
                       source_contract=contract, config=config, deployment=deployment,
                       model_tensor_sha256=digest, runtime_versions=versions(),
                       manifest_sha256=sha(deployment['manifest']), phase='trained_before_validation')
        save_checkpoint(CKPT/'trained_pending_validation.pt', trained)
        validation, report_path = None, None
        improved = ref_improved = False
        if epoch % config['val_every'] == 0 or epoch == config['epochs']:
            write(RUN/'status.json', dict(state='validating', epoch=epoch, completed_epoch=epoch-1,
                epochs=config['epochs'], updates=updates, best=best, pid=os.getpid()))
            validation, report_path = evaluate(model, inputs, rows, config, 'validation', epoch, digest)
            score = validation['metrics'][config['selection_metric']]
            if score is None or not math.isfinite(score): raise FloatingPointError('Invalid checkpoint selection score')
            improved = best is None or score > best['value']
            if improved:
                best = dict(epoch=epoch, metric=config['selection_metric'], value=score,
                            report=report_path, model_tensor_sha256=digest)
            ref_score = validation['metrics']['reference_volume_psnr']
            ref_improved = ref_score is not None and (reference_best is None or ref_score > reference_best['value'])
            if ref_improved:
                reference_best = dict(epoch=epoch, metric='reference_volume_psnr', value=ref_score,
                                      report=report_path, model_tensor_sha256=digest)
        inference = dict(model=model.state_dict(), epoch=epoch, config=config,
                         source_contract=contract, model_tensor_sha256=digest, runtime_versions=versions(), selection=best)
        if improved: save_checkpoint(CKPT/'best.pt', inference)
        if ref_improved: save_checkpoint(CKPT/'best_reference_psnr.pt', dict(inference, selection=reference_best))
        record = dict(epoch=epoch, updates=updates, total_loss=float(np.mean(total)), fm_mse=float(np.mean(fm_values)),
            global_projection_l1_on_active_steps=float(np.mean(global_values)) if global_values else None,
            nonlarge_projection_l1_on_active_steps=float(np.mean(fine_values)) if fine_values else None,
            exact_start_updates=int(mask.sum()), uniform_time_updates=int((~mask).sum()),
            seconds=time.perf_counter()-tick, peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30, best=best,
            validation=validation['metrics'] if validation else None, report=report_path)
        payload = dict(trained, best=best, reference_best=reference_best, phase='epoch_complete', epoch_record=record)
        save_checkpoint(CKPT/'latest.pt', payload)
        if epoch % config['save_every'] == 0 or epoch == config['epochs']:
            save_checkpoint(CKPT/f'epoch_{epoch:04d}.pt', payload)
            if (CKPT/'best.pt').exists(): shutil.copyfile(CKPT/'best.pt', CKPT/f'best_through_{epoch:04d}.pt')
        with (RUN/'history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, allow_nan=False)+'\n')
        inputs.flush()
        write(RUN/'status.json', dict(state='complete' if epoch == config['epochs'] else 'training',
            epoch=epoch, completed_epoch=epoch, epochs=config['epochs'], updates=updates, best=best,
            peak_allocated_gib=record['peak_allocated_gib'], peak_reserved_gib=record['peak_reserved_gib'], pid=os.getpid()))
        print(f'Epoch {epoch} complete; best={best}', flush=True)


def freeze(config, deployment, checkpoint):
    if read(RUN/'status.json')['state'] != 'complete':
        raise RuntimeError('Finish the frozen training budget before selecting the final test checkpoint')
    path = (CKPT/checkpoint).resolve()
    if not path.is_relative_to(CKPT): raise ValueError('Checkpoint must belong to this experiment')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['config'] != config or saved['source_contract'] != source_contract():
        raise ValueError('Checkpoint protocol differs from the delivered experiment')
    if saved['runtime_versions'] != versions(): raise ValueError('Training dependency versions changed before freeze')
    record = dict(checkpoint=path.name, checkpoint_sha256=sha(path), epoch=saved['epoch'],
                  source_contract=source_contract(), config=config, deployment=deployment,
                  runtime_versions=versions(),
                  manifest_sha256=sha(deployment['manifest']),
                  created_at=datetime.now().astimezone().isoformat(), selection='validation only',
                  test_cases_used_for_selection=0)
    destination = ROOT/'verification/final_test_freeze.json'
    if destination.exists():
        old = read(destination)
        if any(old[k] != record[k] for k in ['checkpoint_sha256', 'source_contract', 'config', 'deployment']):
            raise ValueError('Existing final test freeze cannot be repurposed')
        return
    write(destination, record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['bind', 'train', 'resume', 'status', 'evaluate', 'freeze', 'test'])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--labels-root', type=Path)
    parser.add_argument('--checkpoint', default='best.pt', help='Own experiment checkpoint filename')
    args = parser.parse_args()
    config = read(ROOT/'experiment.json')
    if args.action == 'status':
        print(json.dumps(read(RUN/'status.json') if (RUN/'status.json').exists() else
                         dict(state='not_started', epochs=config['epochs']), ensure_ascii=False, indent=2)); return
    delivered = ROOT/'verification/delivery_sources.json'
    if delivered.exists():
        for name, digest in read(delivered).items():
            if sha(ROOT/name) != digest: raise ValueError('Delivered scientific source changed: '+name)
    deployment, rows = binding(args)
    if args.action == 'bind': print(json.dumps(deployment, indent=2)); return
    if args.action in ['train', 'resume']:
        with training_lock(): train(config, deployment, rows, args.action == 'resume')
        return
    if args.action == 'freeze': freeze(config, deployment, args.checkpoint); return
    configure(config)
    checkpoint = CKPT/args.checkpoint
    split = 'validation'
    if args.action == 'test':
        frozen = read(ROOT/'verification/final_test_freeze.json')
        checkpoint = CKPT/frozen['checkpoint']; split = 'test'
        if frozen['source_contract'] != source_contract() or frozen['config'] != config or frozen['deployment'] != deployment:
            raise ValueError('Frozen test protocol changed')
        if frozen['runtime_versions'] != versions(): raise ValueError('Frozen test dependency versions changed')
        if frozen['manifest_sha256'] != sha(deployment['manifest']): raise ValueError('Frozen test manifest changed')
        if sha(checkpoint) != frozen['checkpoint_sha256']: raise ValueError('Frozen checkpoint bytes changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['config'] != config or saved['source_contract'] != source_contract(): raise ValueError('Source/checkpoint mismatch')
    if saved['runtime_versions'] != versions(): raise ValueError('Training/evaluation dependency versions differ')
    model = make_model(16, True, 'voxel').float().to(config['device'])
    model.load_state_dict(saved['model'])
    evaluate(model, Inputs(deployment, config), rows, config, split, saved['epoch'], model_digest(model))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        record = dict(state='failed', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc(), recorded_at=datetime.now().astimezone().isoformat())
        if torch.cuda.is_initialized():
            try:
                free, total = torch.cuda.mem_get_info()
                record['memory'] = dict(free_bytes=free, total_bytes=total,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
            except Exception as memory_error:
                record['memory_query_error'] = str(memory_error)
        write(ROOT/'logs/last_failure.json', record)
        if len(sys.argv) > 1 and sys.argv[1] in ['train', 'resume']:
            previous = read(RUN/'status.json') if (RUN/'status.json').exists() else {}
            if previous.get('pid', os.getpid()) == os.getpid():
                write(RUN/'status.json', dict(previous, state='failed', error=str(exc)))
        raise

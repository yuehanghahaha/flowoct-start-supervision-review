"""Float32 3DFM with TF32 allowed, full-volume evaluation and frozen test."""
import os

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
import platform
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch
from torch.nn import functional as F

from fm3d.data import load_pair, read_rows
from method import make_model, predict
from metrics import PROTOCOL as METRIC_PROTOCOL, quality
from fm3d.inference import generate_volume

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
    paths = sorted(ROOT.glob('*.py')) + [ROOT / 'experiment.json', ROOT / 'membership.json', ROOT / 'requirements.txt']
    paths += sorted((ROOT / 'fm3d').glob('*.py'))
    paths += sorted((ROOT / 'source').rglob('*.py'))
    paths += sorted((ROOT / 'reference').rglob('*.py'))
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
            ['torch', 'torchvision', 'numpy', 'scipy', 'scikit-image', 'Pillow', 'matplotlib', 'torchio', 'monai']}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu']); torch.cuda.set_rng_state_all(state['torch_cuda'])


def restore_best_pointer(selection, filename):
    if selection is None:
        return
    archive = CKPT/selection['archive']
    if not archive.is_file():
        raise FileNotFoundError('Committed best checkpoint archive is missing: '+str(archive))
    saved = torch.load(archive, map_location='cpu', weights_only=False)
    if saved['model_tensor_sha256'] != selection['model_tensor_sha256']:
        raise ValueError('Committed best archive identity mismatch')
    shutil.copyfile(archive, CKPT/filename)


def configure(config, *, device=None):
    expected = dict(tf32=True, gradient_scaling=False,
                    cudnn_benchmark=True, cudnn_deterministic=False, deterministic_algorithms=False)
    for key, value in expected.items():
        if config.get(key) is not value:
            raise ValueError(f'Frozen precision setting {key} must be {value!r}')
    device = torch.device(config['device'] if device is None else device)
    if device.type == 'cuda':
        if os.getenv('NVIDIA_TF32_OVERRIDE') == '0':
            raise RuntimeError('Unset NVIDIA_TF32_OVERRIDE=0 before running this TF32-enabled experiment')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for the configured experiment device')
    if config['amp'] and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Configured BF16 requires native GPU support')
    if config['autocast_dtype'] != 'bfloat16': raise ValueError('Only declared BF16 is supported')
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(config['threads'])
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = config['tf32']
    torch.backends.cudnn.allow_tf32 = config['tf32']
    torch.backends.cudnn.benchmark = config['cudnn_benchmark']
    torch.backends.cudnn.deterministic = config['cudnn_deterministic']
    torch.use_deterministic_algorithms(config['deterministic_algorithms'])


def precision_flags(model=None):
    dtype = torch.get_default_dtype() if model is None else next(model.parameters()).dtype
    cfg = read(ROOT/'experiment.json')
    return dict(tensor_dtype=str(dtype), model_parameter_dtype=str(dtype),
                autocast_configured=cfg['amp'], autocast_dtype=cfg['autocast_dtype'],
                loss_and_ode_state_dtype='float32',
                autocast_enabled_at_recording=torch.is_autocast_enabled(),
                gradient_scaling=False, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                environment={name: os.getenv(name) for name in
                             ('NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')})


@contextmanager
def training_lock():
    if os.name != 'posix':
        raise RuntimeError('Use the Linux A100 environment for formal training')
    import fcntl
    path = ROOT / 'runs/operation.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+', encoding='utf-8') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This experiment already has an active training/evaluation process')
        stream.seek(0); stream.truncate(); stream.write(str(os.getpid())); stream.flush()
        yield


def binding(args):
    path = ROOT / 'deployment.json'
    if args.data_root or args.manifest or args.labels_root or args.cache_root or not path.exists():
        data = Path(args.data_root or '/docsystem/niuke/lvyuehang/OCTA500_common_v1').resolve()
        manifest = args.manifest or data/'split_3M_z256.csv'
        value = dict(data_root=str(data), manifest=str(Path(manifest).resolve()),
                     labels_root=str(Path(args.labels_root or ROOT / 'annotations').resolve()),
                     cache_root=str(Path(args.cache_root or '/docsystem/niuke/lvyuehang/OCTA500_research_cache_3M_z256_v1').resolve()))
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
        names = ['GT_Layers', 'GT_Capillary', 'GT_LargeVessel'] if row['split'] == 'validation' else (['GT_Layers'] if read(ROOT/'experiment.json')['projection_mode'] != 'none' else [])
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
        self.cache = Path(deployment['cache_root'])
        self.cache_info = read(self.cache/'cache.json')
        self.identify(self.cache/'cache.json')
        expected = {r['case_id']: r['split'] for r in read(ROOT/'membership.json') if r['split'] != 'test'}
        if {cid: v['split'] for cid, v in self.cache_info['cases'].items()} != expected:
            raise ValueError('Cache must match the fixed 140 training / 10 validation patients')
        for key, value in dict(axis_order='ZYX', depth=256, dtype='float32', intensity_scale=255., fov='3M').items():
            if self.cache_info['protocol'][key] != value:
                raise ValueError('Cache preprocessing differs: ' + key)

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
        if row['split'] == 'test':
            source, target = load_pair(row['path'], 'ZYX', 255., self.config['depth'])
        else:
            cid = row['case_id']
            original_digest = self.identities[str(Path(row['path']).resolve())]['sha256']
            if original_digest != self.cache_info['cases'][cid]['source_npz_sha256']:
                raise ValueError('Cache/source identity mismatch: ' + cid)
            arrays = []
            for key in ['oct', 'octa']:
                path = self.cache/cid/(key+'.npy'); self.identify(path)
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                if array.shape != (1, 256, 304, 304) or array.dtype != np.float32:
                    raise ValueError('Invalid existing read-only cache: ' + str(path))
                arrays.append(array)
            source, target = arrays
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
        return (torch.from_numpy(np.array(source[region], copy=True))[None],
                torch.from_numpy(np.array(target[region], copy=True))[None], origin, slices)


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
    # Every evaluated object has a complete MAE/PSNR/SSIM set in this table.
    standards = [key for key in keys if key.endswith(('_mae', '_psnr', '_ssim'))]
    table = ''.join(f'<tr><td>{html.escape(k)}</td><td>{result["metrics"][k]}</td></tr>' for k in standards)
    cases = ''.join(f'<h2>{r["case_id"]}</h2><img loading="lazy" src="figures/{r["case_id"]}.png">' for r in rows)
    flags = result['identity']['precision_flags']
    precision = (f'{flags["tensor_dtype"]}, matmul TF32={flags["matmul_allow_tf32"]}, '
                 f'cuDNN TF32={flags["cudnn_allow_tf32"]}')
    doc = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Baseline evaluation</title>'
           '<style>body{font:16px/1.7 sans-serif;max-width:1150px;margin:35px auto;padding:20px;color:#203342}'
           'img{max-width:100%}td{border:1px solid #ccd7df;padding:8px}table{border-collapse:collapse}</style>'
           f'<h1>{html.escape(result["experiment_id"])} / {result["split"]}</h1>'
           f'<p>Epoch {result["epoch"]}, patients={result["patients"]}, {precision}, {html.escape(result["config"]["inference_description"])}, full 256x304x304.</p>'
           '<p>Volume MAE/PSNR/3D SSIM: fixed range=1. Reference columns: per-patient GT max-min. '
           'SSIM windows=7. Equal-patient aggregation. B-scan=all Y slices [Z,X], slice then volume then patient. '
           'Main projection=ILM-OPL maximum. XOCT-style slab means and GT-normalized uint8 arithmetic are separate supplements. '
           'Region errors are diagnostics, not vessel-weighted TransPro metrics.</p>'
           '<p><a href="per_case_metrics.csv">All case metrics</a> | <a href="metrics.json">Complete definitions/results</a></p>'
           '<table>'+table+'</table>'+cases+'</html>')
    (folder / 'index.html').write_text(doc, encoding='utf-8')


@torch.inference_mode()
def evaluate(model, inputs, rows, config, split, epoch, weights_digest):
    import csv
    if split not in ('validation', 'test'):
        raise ValueError('Only fixed validation/test splits can be evaluated')
    selected = [r for r in rows if r['split'] == split]
    expected = 10 if split == 'validation' else 50
    if len(selected) != expected: raise ValueError('Invalid evaluation patient count')
    identity = dict(experiment_id=config['experiment_id'], split=split, epoch=epoch,
                    model_tensor_sha256=weights_digest, source_contract=source_contract(), runtime_versions=versions(), inference_protocol=config['inference_description'],
                    metric_protocol=METRIC_PROTOCOL, precision_flags=precision_flags(model), keep_volume=(split == 'test'))
    tag = f'epoch_{epoch:04d}_{weights_digest[:12]}'
    folder = ROOT / 'reports' / split / tag
    output = ROOT / 'predictions' / split / tag
    folder.mkdir(parents=True, exist_ok=True)
    keep_volume = split == 'test'
    if keep_volume: output.mkdir(parents=True, exist_ok=True)
    (folder / 'figures').mkdir(exist_ok=True); (folder / 'views').mkdir(exist_ok=True)
    if (folder / 'identity.json').exists() and read(folder / 'identity.json') != identity:
        raise ValueError('Evaluation identity mismatch; refusing to reuse output')
    write(folder / 'identity.json', identity)
    records = []
    for i, row in enumerate(selected):
        cid = row['case_id']; record_path = folder / (cid+'.json'); prediction_path = output / (cid+'.npz')
        inputs.identify(row['path']); layers = inputs.layers(cid); regions = inputs.regions_xy(cid)
        input_digest = inputs.identities[str(Path(row['path']).resolve())]['sha256']
        dependencies = {str(Path(row['path']).resolve()): input_digest}
        for group, suffix in [('GT_Layers', '.mat'), ('GT_Capillary', '.bmp'), ('GT_LargeVessel', '.bmp')]:
            name = str((inputs.labels/group/(cid+suffix)).resolve())
            dependencies[name] = inputs.identities[name]['sha256']
        if record_path.exists():
            record = read(record_path)
            if record['input_identities'] != dependencies:
                raise ValueError('Saved result/source checksum mismatch')
            if keep_volume and (not prediction_path.is_file() or sha(prediction_path) != record['prediction_sha256']):
                raise ValueError('Formal test prediction missing or changed')
            for name, digest in record['report_files'].items():
                if not (folder/name).is_file() or sha(folder/name) != digest:
                    raise ValueError('Saved evaluation evidence missing or changed: '+name)
            records.append(record); continue
        tick = time.perf_counter()
        source, target = inputs.pair(row)
        source = source if torch.is_tensor(source) else torch.from_numpy(np.array(source, copy=True))
        target = target if torch.is_tensor(target) else torch.from_numpy(np.array(target, copy=True))
        if config['method'] == 'xoct_plain':
            with np.load(row['path'], allow_pickle=False) as raw_input:
                source = torch.from_numpy(raw_input['oct'].astype(np.float64)/255)[None]
        prediction = predict(model, source[None], config)
        pred, gt = prediction[0, 0].numpy(), target[0].numpy()
        if pred.dtype != np.float32 or not np.isfinite(pred).all():
            raise FloatingPointError('Invalid full-resolution float32 prediction')
        if keep_volume:
            temporary = prediction_path.with_name(cid+'.tmp.npz')
            np.savez_compressed(temporary, prediction=pred, axis_order=np.array('ZYX'),
                                case_id=np.array(cid), epoch=np.int32(epoch),
                                model_tensor_sha256=np.array(weights_digest), input_sha256=np.array(input_digest))
            temporary.replace(prediction_path)
        metrics, p, g, bscans = quality(pred, gt, layers, regions, include_reference=keep_volume)
        np.savez_compressed(folder/'views'/f'{cid}.npz',
                            **{'pred_'+k: v for k, v in p.items()}, **{'gt_'+k: v for k, v in g.items()})
        native_protocol = None
        case_figure(folder/'figures'/f'{cid}.png', cid, p, g, pred, gt)
        with (folder/'views'/f'{cid}_bscans.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['y', 'mae', 'mse', 'psnr', 'ssim'])
            writer.writeheader(); writer.writerows(bscans)
        # Fixed, predeclared slice positions; no selection by test quality.
        positions = [0, 76, 152, 228, 303]
        np.savez_compressed(folder/'views'/f'{cid}_slices.npz', y=np.array(positions),
                            oct=source[0].numpy()[:, positions, :].transpose(1, 0, 2),
                            gt=gt[:, positions, :].transpose(1, 0, 2),
                            prediction=pred[:, positions, :].transpose(1, 0, 2))
        evidence = [f'views/{cid}.npz', f'views/{cid}_bscans.csv', f'views/{cid}_slices.npz', f'figures/{cid}.png']
        record = dict(case_id=cid, patient_id=row['patient_id'], metrics=metrics,
                      input_sha256=input_digest, input_identities=dependencies,
                      prediction_sha256=sha(prediction_path) if keep_volume else None,
                      full_prediction=str(prediction_path.relative_to(ROOT)) if keep_volume else None,
                      prediction_retention='formal_test_full_float32' if keep_volume else '3D arrays released after metrics; 2D evidence retained',
                      native_reference_protocol=native_protocol if keep_volume else None,
                      report_files={name: sha(folder/name) for name in evidence}, shape_ZYX=list(pred.shape),
                      model_tensor_sha256=weights_digest, seconds=time.perf_counter()-tick)
        write(record_path, record); records.append(record)
        inputs.flush()
        del prediction, pred, gt, source, target, p, g
        print(f'{split} {i+1}/{len(selected)} {cid}: volume MAE={metrics["volume_mae"]:.6f}, '
              f'PSNR={metrics["volume_psnr"]:.3f}, 3D SSIM={metrics["volume_ssim"]:.4f}', flush=True)
    result = dict(experiment_id=config['experiment_id'], split=split, epoch=epoch,
                  model_tensor_sha256=weights_digest, **patient_summary(records), config=config,
                  identity=identity, protocol=METRIC_PROTOCOL,
                  full_prediction_policy='all50 formal test volumes retained' if keep_volume else 'no intermediate validation3D files')
    write(folder/'metrics.json', result); evaluation_report(folder, result)
    write(ROOT/'reports'/split/'latest_report.json', dict(folder=str(folder.relative_to(ROOT)), epoch=epoch))
    return result, str(folder.relative_to(ROOT))


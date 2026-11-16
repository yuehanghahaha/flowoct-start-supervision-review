"""Experiment paths, reproducibility, inputs and evaluation rendering."""
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image
from scipy.io import loadmat
import torch

from fm3d.data import read_rows

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/train"
CKPT = ROOT / "checkpoints"
VERIFY = ROOT / "verification"


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


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch_cpu=torch.get_rng_state(), torch_cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu']); torch.cuda.set_rng_state_all(state['torch_cuda'])


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
                optimizer_state_dtype='float32',
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
        raise RuntimeError('Use the Linux CUDA environment for formal training')
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


def binding():
    value = read(ROOT / 'deployment.json')
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
        names = ['GT_Layers', 'GT_Capillary', 'GT_LargeVessel'] if row['split'] == 'validation' else []
        for name in names:
            suffix = '.mat' if name == 'GT_Layers' else '.bmp'
            required = Path(value['labels_root'])/name/(row['case_id']+suffix)
            if not required.is_file(): raise FileNotFoundError(f'Missing required annotation: {required}; use --labels-root')
    return value, rows


class Inputs:
    def __init__(self, deployment, config, verified=None):
        self.labels = Path(deployment['labels_root'])
        self.config = config
        self.identity_path = VERIFY / 'input_identity.json'
        self.identities = dict(verified or {})
        self.seen = set(self.identities)
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
        return self.identities[name]['sha256']

    def flush(self):
        write(self.identity_path, self.identities)

    def pair(self, row):
        if row['split'] not in ('train', 'validation'):
            raise ValueError('Only fixed training and validation inputs are allowed')
        cid = row['case_id']
        original_digest = self.identify(row['path'])
        if original_digest != self.cache_info['cases'][cid]['source_npz_sha256']:
            raise ValueError('Cache/source identity mismatch: ' + cid)
        arrays = []
        for key in ('oct', 'octa'):
            path = self.cache / cid / (key + '.npy')
            self.identify(path)
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.shape != (1, 256, 304, 304) or array.dtype != np.float32:
                raise ValueError('Invalid existing read-only cache: ' + str(path))
            arrays.append(array)
        return tuple(arrays)

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
        if row['split'] != 'train':
            raise ValueError('Training crops require a training patient')
        source, target = self.pair(row)
        # Match the parent's Python-random crop order, independent of time/loss RNG.
        origin = tuple(random.randint(0, n-p) for n, p in zip(source.shape[1:], self.config['patch']))
        slices = tuple(slice(s, s+p) for s, p in zip(origin, self.config['patch']))
        region = (slice(None),) + slices
        return (torch.from_numpy(np.array(source[region], copy=True))[None],
                torch.from_numpy(np.array(target[region], copy=True))[None], origin)


def time_schedule(config, epoch):
    rng = np.random.default_rng(config['seed'] + epoch + 100000)
    times = rng.random(config['steps_per_epoch']).astype(np.float32)
    times = np.clip(times, np.nextafter(np.float32(0), np.float32(1)),
                    np.nextafter(np.float32(1), np.float32(0)))
    return times


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
    first = html.escape(rows[0]['case_id'])
    cases = f'<h2>{first}</h2><img loading="lazy" src="figures/{first}.png">'
    flags = result['identity']['precision_flags']
    precision = (f'{flags["tensor_dtype"]}, matmul TF32={flags["matmul_allow_tf32"]}, '
                 f'cuDNN TF32={flags["cudnn_allow_tf32"]}')
    doc = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>OCT-start validation</title>'
           '<style>body{font:16px/1.7 sans-serif;max-width:1150px;margin:35px auto;padding:20px;color:#203342}'
           'img{max-width:100%}td{border:1px solid #ccd7df;padding:8px}table{border-collapse:collapse}</style>'
           f'<h1>{html.escape(result["experiment_id"])} / {result["split"]}</h1>'
           f'<p>Epoch {result["epoch"]}, patients={result["patients"]}, {precision}, {html.escape(result["config"]["inference_description"])}, full 256x304x304.</p>'
           '<p>Volume MAE/PSNR/3D SSIM: fixed range=1. Reference columns: per-patient GT max-min. '
           'SSIM windows=7. Equal-patient aggregation. B-scan=all Y slices [Z,X], slice then volume then patient. '
           'Main projection=ILM-OPL maximum. Slab means use the separate float32 range-1 protocol. '
           'Region errors are supplementary diagnostics.</p>'
           '<p><a href="per_case_metrics.csv">All case metrics</a> | <a href="metrics.json">Complete definitions/results</a></p>'
           '<table>'+table+'</table>'+cases+'</html>')
    (folder / 'index.html').write_text(doc, encoding='utf-8')

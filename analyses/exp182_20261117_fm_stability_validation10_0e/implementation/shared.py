"""Frozen validation inputs, original networks, and atomic evidence records."""
import csv
import hashlib
import importlib.metadata
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat

from fm3d.data import load_pair
from fm3d.model import make_model
from source.xoct.hybrid_generator import UNet3D2DAll
from source.xoct.msff3d_generator import UNet3DMSFF

ROOT = Path(__file__).resolve().parent
OWNED = Path('/docsystem/niuke/lvyuehang').resolve()
BASE = OWNED / 'Flow_oct2octa/experience'
DATA = OWNED / 'OCTA500_common_v1'
ANNOT = BASE / 'exp102_20261106_start_alltproj_150e/annotations'
XOCT = BASE / 'exp059_20261102_xoct_paperlr_300e'
HPC_PARENT = '/home/lvyuehang/experience/exp163_20261114_startproj_layermean_a100_200e'
GPU_UUID = 'GPU-58b67969-49a6-611a-807b-e256f9ec4c31'


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def read(path):
    return Path(path).read_text(encoding='utf-8-sig')


def config():
    return json.loads(read(ROOT / 'experiment.json'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bounded(path, base=OWNED):
    path = Path(path).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f'Path outside experiment inputs: {path}')
    return path


def save_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, array, allow_pickle=False)
    temporary.replace(path)


def save_views(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def tensor_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def model_weights(tag, config_value, frozen=None):
    model_cfg = config_value['models'][tag]
    path = ROOT / 'checkpoints' / model_cfg['file']
    if frozen is not None and sha(path) != frozen['models'][tag]['sha256']:
        raise ValueError(f'Frozen checkpoint changed: {tag}')
    if tag == 'XOCT_E213':
        model = UNet3D2DAll(UNet3DMSFF(1, 1, 64, 512, 'group', True, True), 64, 'group')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        weights = checkpoint['model']
    else:
        model = make_model(base_channels=16, condition=True)
        # Trusted project checkpoints also contain optimizer and NumPy RNG state.
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        weights = checkpoint['model']
    if checkpoint['epoch'] != model_cfg['epoch']:
        raise ValueError(f'Unexpected saved epoch: {tag}')
    model.load_state_dict(weights, strict=True)
    identity = tensor_digest(model)
    if frozen is not None and frozen['models'][tag]['tensor_sha256'] != identity:
        raise ValueError(f'Frozen tensor identity changed: {tag}')
    model.eval()
    return model, identity


def validation_members(cfg):
    if sha(ROOT / 'membership.json') != cfg['fixed_membership_sha256']:
        raise ValueError('Original patient membership changed')
    original = json.loads(read(ROOT / 'membership.json'))
    keys = ['case_id', 'patient_id', 'split']
    with bounded(DATA / 'split_3M_z256.csv').open(encoding='utf-8-sig', newline='') as stream:
        manifest = [{k: row[k] for k in keys} for row in csv.DictReader(stream)]
    if sorted(original, key=lambda r: r['case_id']) != sorted(manifest, key=lambda r: r['case_id']):
        raise ValueError('A800 patient membership differs from exp163')
    rows = [r for r in original if r['split'] == 'validation']
    if [r['case_id'] for r in rows] != cfg['validation_cases'] or len({r['patient_id'] for r in rows}) != 10:
        raise ValueError('Fixed validation patients do not match configuration')
    return {r['case_id']: r for r in rows}


def input_paths(cid):
    result = dict(
        pair=bounded(DATA / 'z256/3M' / f'{cid}.npz'),
        layers=bounded(ANNOT / 'GT_Layers' / f'{cid}.mat'),
        capillary=bounded(ANNOT / 'GT_Capillary' / f'{cid}.bmp'),
        large=bounded(ANNOT / 'GT_LargeVessel' / f'{cid}.bmp'),
    )
    for key, name, extension in [('faz2d', 'GT_FAZ', '.bmp'), ('faz3d', 'GT_FAZ3D_Z256', '.npz')]:
        source = ROOT / 'source/annotations' / name / (cid + extension)
        if source.is_file():
            result[key] = bounded(source, ROOT)
    return result


def binary_mask(path, shape):
    with Image.open(path) as im:
        mask = np.asarray(im).copy()
    if mask.shape != shape or not set(np.unique(mask)).issubset({0, 1, 255}):
        raise ValueError(f'Expected binary bitmap {path}')
    return mask > 0


def inputs(cid, cfg, frozen):
    shape = tuple(cfg['shape'])
    paths = input_paths(cid)
    digests = {key: sha(path) for key, path in paths.items()}
    if digests != frozen['inputs'][cid]['sha256']:
        raise ValueError(f'Validation file identity changed for {cid}')
    source, target = load_pair(paths['pair'], 'ZYX', 255., shape[0])
    if tuple(source.shape) != (1, *shape) or source.shape != target.shape:
        raise ValueError(f'Full-resolution geometry mismatch for {cid}')
    layers = np.asarray(loadmat(paths['layers'])['Layer'], dtype=np.float32)
    if (layers.shape != (6, *shape[1:]) or not np.isfinite(layers).all() or
        layers.min() < 1 or layers.max() > 640 or (np.diff(layers, axis=0) < 0).any()):
        raise ValueError(f'Invalid layer geometry for {cid}')
    labels = {key: binary_mask(paths[key], shape[1:]) for key in ('capillary', 'large', 'faz2d') if key in paths}
    if 'faz3d' in paths:
        with np.load(paths['faz3d'], allow_pickle=False) as result:
            packed = result['packed_ZYX']
            labels['faz3d'] = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)
    # Source annotations are XY; projections in this experiment are XY. Volume masks transpose to YX.
    return source[None], target[None], layers, labels


def configure_gpu():
    if os.getenv('NVIDIA_TF32_OVERRIDE') == '0' or not torch.cuda.is_available():
        raise RuntimeError('Existing A800 CUDA and the declared TF32 policy are required')
    if torch.cuda.device_count() != 1 or 'A800' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Expected one visible A800 device')
    device = torch.device('cuda:0')
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    return device


def runtime():
    return dict(
        gpu=torch.cuda.get_device_name(0), torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        versions={name: importlib.metadata.version(name) for name in
                  ['torch', 'numpy', 'scipy', 'scikit-image', 'matplotlib', 'Pillow']},
        dtype=str(torch.get_default_dtype()), tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32, amp=torch.is_autocast_enabled(), scaler=False,
        cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cuda_visible_devices=os.getenv('CUDA_VISIBLE_DEVICES'),
    )


def commit(folder, record, files):
    files = {path.relative_to(ROOT).as_posix(): sha(path) for path in files}
    write(folder / 'record.json', dict(**record, files=files, committed=now()))


def existing(folder, identity, verify=True):
    path = folder / 'record.json'
    if not path.is_file():
        return None
    record = json.loads(read(path))
    if record['identity'] != identity:
        raise ValueError(f'Resume identity mismatch: {folder}')
    if verify:
        for relative, digest in record['files'].items():
            file_path = bounded(ROOT / relative, ROOT)
            if not file_path.is_file() or sha(file_path) != digest:
                raise ValueError(f'Missing or changed committed evidence: {file_path}')
    return record


def status(phase, state, **extra):
    write(ROOT / f'runs/{phase}.json', dict(state=state, time=now(), pid=os.getpid(), **extra))


def elapsed(tick):
    return round(time.perf_counter() - tick, 3)

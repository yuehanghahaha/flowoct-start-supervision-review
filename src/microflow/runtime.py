from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
import random
import numpy as np
import torch


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024**2), b''): h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


@contextmanager
def atomic(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        yield tmp
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def write(path, obj):
    with atomic(path) as p:
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def save(path, obj):
    with atomic(path) as p, p.open('wb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def seed_for(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:8], 'little') % (2**63)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed)


def rng_state(device):
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state(device) if device.type=='cuda' else None}


def restore_rng(state, device):
    torch.set_rng_state(state['torch'])
    if device.type=='cuda': torch.cuda.set_rng_state(state['cuda'], device)


def configure(cfg, device, run):
    profile=cfg['precision']
    if profile['parameters']!='float32' or profile['network_autocast']!='bfloat16' or profile['state_and_loss']!='float32' or profile['scaler']:
        raise ValueError('Declare the implemented BF16 autocast/FP32 state protocol accurately; no silent precision changes.')
    run = Path(run).resolve(); run.mkdir(parents=True, exist_ok=True)
    if os.name != 'nt':
        allowed = Path.home().resolve()
        if allowed not in run.parents: raise ValueError('A100 outputs must stay in the authorized persistent home.')
    for key in ['TMPDIR', 'XDG_CACHE_HOME', 'TORCH_HOME', 'TRITON_CACHE_DIR', 'CUDA_CACHE_PATH']:
        p = run/'cache'/key.lower(); p.mkdir(parents=True, exist_ok=True); os.environ[key] = str(p)
    torch.set_num_threads(4)
    if device.type=='cuda':
        if cfg.get('execution',{}).get('require_hpc_container',True):
            import getpass,socket
            home=Path.home().resolve();user=getpass.getuser()
            if home!=Path('/workspace') or socket.gethostname() not in (f'hpc-{user}-slot1',f'hpc-{user}-slot2'):
                raise RuntimeError('Enter your own allocated hpc-gpu slot container before CUDA work.')
            receipt=read(run/'platform_mapping.json')
            observed=home.stat()
            if receipt['account']!=user or receipt['persistent_home_inode']!=[observed.st_dev,observed.st_ino]:
                raise RuntimeError('Persistent host-home mapping does not match the registered container.')
        if torch.cuda.device_count()!=1:raise RuntimeError('Preserve the platform single-GPU visibility.')
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported(): raise RuntimeError('Declared BF16 needs supported hardware.')
        torch.backends.cuda.matmul.allow_tf32=profile['tf32_matmul']
        torch.backends.cudnn.allow_tf32=profile['tf32_cudnn']
        torch.backends.cudnn.benchmark=profile['benchmark']
        torch.backends.cudnn.deterministic=profile['deterministic']
        torch.use_deterministic_algorithms(False)
    from importlib.metadata import version,PackageNotFoundError
    packages={}
    for name in ('torch','numpy','scipy','scikit-image','Pillow','mamba-ssm','causal-conv1d'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    return {'time':stamp(), 'device':str(device), 'torch':str(torch.__version__), 'precision':cfg['precision'],
            'packages':packages,'cuda':torch.version.cuda,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn':torch.backends.cudnn.version() if device.type=='cuda' else None,
            'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None}


@contextmanager
def amp(device):
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type=='cuda'): yield


def source_signature(root, config, manifest):
    root=Path(root)
    files=list((root/'src').rglob('*.py'))+[root/'run.py', Path(config)]
    return {'sources':{str(p.relative_to(root)):digest(p) for p in sorted(set(files))},
            'manifest_sha256':digest(manifest)}


class Pause(Exception): pass


def paused(run): return (Path(run)/'pause.request').exists()


@contextmanager
def operation_lock(run):
    p=Path(run)/'operation.lock';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+b') as f:
        if os.name=='nt':
            import msvcrt
            if p.stat().st_size==0:f.write(b'0');f.flush()
            f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

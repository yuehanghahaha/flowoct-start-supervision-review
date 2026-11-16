"""Experiment paths, atomic writes, reproducibility, and device settings."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform

import torch


ROOT = Path(__file__).resolve().parent
PAUSE_EXIT = 75


class PauseRequested(Exception):
    def __init__(self, phase, **progress):
        super().__init__(f"Paused at a committed {phase} boundary.")
        self.progress = dict(phase=phase, **progress)


def pause_requested(root):
    return (root / "runs/pause.request").is_file()


def pause_at_boundary(root, phase, **progress):
    if pause_requested(root):
        raise PauseRequested(phase, **progress)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


@contextmanager
def atomic_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_checkpoint(path, payload):
    with atomic_path(path) as temporary:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*fields):
    packed = json.dumps(fields, separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(packed).digest()[:8], "little") % (2**63)


def source_contract(root=ROOT):
    files = [root / name for name in ("experiment.json", "membership.json", "requirements.txt",
                                      "run.py", "runtime.py", "dataset.py", "metrics.py", "report.py", "train.py",
                                      "a100.py", "return_results.py", "normalization.py", "diagnostics.py", "runs/slot2/control.py", "verify.py", "verification/reference_scan.py")]
    files.extend(sorted((root / "fm3d").glob("*.py")))
    return {path.relative_to(root).as_posix(): sha256(path) for path in files}


def guard_path(path):
    resolved = Path(path).resolve()
    if os.name != 'nt':
        allowed = Path('/workspace') if ROOT.parts[1] == 'workspace' else Path('/home/HPC_USER')
        if resolved != allowed and allowed not in resolved.parents:
            raise ValueError(f'Path escapes the authorized user directory: {path} -> {resolved}')
    return resolved


def load_config(root=ROOT):
    config = read_json(root / 'experiment.json')
    if config['full_shape'] != [256, 304, 304] or config['batch_size'] != 2:
        raise ValueError('This protocol requires two full-resolution patients per update.')
    if config['epochs'] != 200 or config['steps_per_epoch'] != 200:
        raise ValueError('The declared budget is 40000 complete optimizer updates.')
    return config


def configure(config, device):
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('The formal experiment requires the allocated CUDA Mamba GPU.')
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Declared BF16 autocast is unsupported on this GPU.')
    torch.set_num_threads(config['cpu_threads'])
    profile = config['precision']
    torch.backends.cuda.matmul.allow_tf32 = profile['tf32_matmul']
    torch.backends.cudnn.allow_tf32 = profile['tf32_cudnn']
    torch.backends.cudnn.benchmark = profile['cudnn_benchmark']
    torch.backends.cudnn.deterministic = profile['cudnn_deterministic']
    torch.use_deterministic_algorithms(False)
    if not profile['amp'] or profile['network_autocast_dtype'] != 'bfloat16' or profile['gradient_scaling']:
        raise ValueError('This experiment uses BF16 autocast without gradient scaling.')
    props = torch.cuda.get_device_properties(device)
    return dict(device=str(device), gpu=props.name, memory_bytes=props.total_memory,
                gpu_uuid=str(getattr(props, 'uuid', '')), python=platform.python_version(),
                platform=platform.platform(), cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                deterministic_algorithms=False, precision=dict(profile),
                packages={name: version(name) for name in
                          ('torch', 'numpy', 'scipy', 'scikit-image', 'Pillow', 'mamba-ssm', 'causal-conv1d')})


@contextmanager
def operation_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        try:
            if os.name == "nt":
                import msvcrt
                if stream.tell() == 0:
                    stream.write(b"0")
                    stream.flush()
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another operation is writing to {path.parent}.") from error
        yield

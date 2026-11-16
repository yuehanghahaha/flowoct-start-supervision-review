"""Local experiment utilities; all generated artifacts stay in this directory."""
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def write(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Generated records must stay within this experiment")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def identity():
    paths = sorted([ROOT / "experiment.json", ROOT / "data/membership.json", *ROOT.glob("*.py"), *ROOT.glob("source/**/*.py")])
    hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    for name in ("hcg", "vpg"):
        hashes["pretrain_weights/" + name + ".pth"] = sha(ROOT / "pretrain_weights" / (name + ".pth"))
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def precision(config, seed=None):
    if config["amp"] or config["gradient_scaling"] or config["dtype"] != "float32":
        raise ValueError("This comparison keeps float32 tensors with AMP/scaling disabled")
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = config["matmul_tf32"]
    torch.backends.cudnn.allow_tf32 = config["tf32"]
    torch.backends.cudnn.benchmark = config["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = config["cudnn_deterministic"]
    torch.set_num_threads(config["torch_threads"])
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def checkpoint(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / "checkpoints"):
        raise ValueError("Checkpoint path outside this experiment")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def runtime(device):
    return {"time_unix": time.time(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "device": str(device), "gpu": torch.cuda.get_device_name(device),
            "gpu_total_bytes": torch.cuda.get_device_properties(device).total_memory,
            "gpu_free_bytes": torch.cuda.mem_get_info(device)[0],
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "float_tensor_dtype": str(torch.get_default_dtype()),
            "amp": False, "gradient_scaling": False}


def append_event(event):
    path = ROOT / "logs/events.jsonl"
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(safe({"time_unix": time.time(), **event}), ensure_ascii=False) + "\n")

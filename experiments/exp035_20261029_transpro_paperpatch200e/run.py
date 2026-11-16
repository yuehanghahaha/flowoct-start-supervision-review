"""Public TransPro training with documented paper loss patches and server integration."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import time
import traceback

import numpy as np
import torch

from experiment import ROOT, read, write, sha, identity, precision, runtime, append_event
from dataset import Inputs, prepare

# Optional experiment-local dependencies; never replace a running experiment's environment.
if (ROOT / "runtime_deps").is_dir():
    sys.path.insert(0, str(ROOT / "runtime_deps"))
sys.path.insert(0, str(ROOT / "source"))


def protocol_id():
    return hashlib.sha256((identity() + sha(ROOT / "binding.json")).encode()).hexdigest()


def author_argv(gpu="0"):
    return [str(ROOT / "train_author.py"), "--dataroot", read(ROOT / "binding.json")["cache"],
            "--name", "transpro_3M", "--model", "TransPro", "--netG", "unet_256", "--direction", "AtoB",
            "--lambda_A", "10", "--lambda_C", "5", "--dataset_mode", "alignedoct2octa3d", "--norm", "batch",
            "--pool_size", "0", "--load_size", "304", "--input_nc", "1", "--output_nc", "1", "--gpu_ids", gpu,
            "--gan_mode", "paper_logarithmic_minimax",
            "--no_flip", "--num_threads", "4", "--checkpoints_dir", str(ROOT / "checkpoints/author_log"),
            "--display_id", "-1", "--no_html"]


@contextmanager
def exclusive_run():
    if sys.platform != "linux":
        raise RuntimeError("Formal training and evaluation run on the Linux GPU server")
    import fcntl
    (ROOT / "runs").mkdir(exist_ok=True)
    with (ROOT / "runs/train.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def output_budget(trained_bytes, teacher_bytes):
    # Resume contains networks, Adam, teacher buffers and the selected best weights.
    resume = 4 * trained_bytes + teacher_bytes
    predictions = 60 * 256 * 304 * 304 * 4
    numbered = 200 * trained_bytes  # Worst case: every epoch improves best; same-epoch saves overwrite as in the author loop.
    phases = {"atomic_resume": numbered + 2 * resume + 2 * trained_bytes,
              "final_outputs": numbered + resume + 2 * trained_bytes + predictions}
    return {"required_bytes": max(phases.values()) + 10 * 1024**3, "phase_bytes": phases,
            "margin_bytes": 10 * 1024**3, "numbered_epoch_sets":200,
            "scope":"Author numbered/best/latest weights, resume and 60 final predictions; first complete run, existing replay attempts excluded"}


def check(config, gpu):
    from options.train_options import TrainOptions
    from models import create_model
    precision(config)
    Inputs(read(ROOT / "binding.json"))
    result = {"protocol_id": protocol_id(), "synthetic_only": True, "training_started": False}
    old_argv = sys.argv
    try:
        sys.argv = author_argv(gpu)
        opt = TrainOptions().parse()
        model = create_model(opt)
        model.setup(opt)
        device = torch.device("cuda:" + gpu)
        result["runtime"] = {**runtime(device), "cudnn_benchmark": torch.backends.cudnn.benchmark,
                             "cudnn_deterministic": torch.backends.cudnn.deterministic,
                             "precision_policy": "author_framework_defaults; float32 tensors; TF32 not forced off"}
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        a = torch.rand((1, 1, 256, 256, 256), device=device) * 2 - 1
        b = torch.rand_like(a) * 2 - 1
        model.set_input({"A": a, "B": b, "A_paths": ["SYNTHETIC"]})
        model.optimize_parameters()  # Released update order, with only the documented paper loss patch.
        losses = model.get_current_errors()
        torch.cuda.synchronize(device)
        if not all(np.isfinite(value) for value in losses.values()):
            raise FloatingPointError("Non-finite synthetic author-model update")
        trained = sum(v.numel() * v.element_size() for name in ("netG", "netD", "netD_proj") for v in getattr(model, name).state_dict().values())
        teachers = sum(v.numel() * v.element_size() for name in ("netG_t", "net") for v in getattr(model, name).state_dict().values())
        budget = output_budget(trained, teachers)
        free = shutil.disk_usage(ROOT).free
        result.update(losses=losses, gpu_step_passed=True, seconds=time.perf_counter()-started,
                      max_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      max_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      storage_budget=budget, filesystem_free_bytes=free)
        if free < budget["required_bytes"]:
            raise RuntimeError(f"Measured free space {free / 1024**3:.2f} GiB; output budget {budget['required_bytes'] / 1024**3:.2f} GiB")
        result["status"] = "passed"
    except BaseException as exc:
        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        sys.argv = old_argv
        write(ROOT / "verification/gpu_check.json", result)
    print(json.dumps(result, indent=2))
    print("Paper-patched TransPro GPU check passed. Formal training has NOT started.", flush=True)


def train(config, gpu):
    accepted = read(ROOT / "verification/gpu_check.json")
    if accepted["status"] != "passed" or accepted["protocol_id"] != protocol_id():
        raise ValueError("Run this experiment's full-size GPU check first")
    precision(config)
    old_argv = sys.argv
    try:
        sys.argv = author_argv(gpu)
        runpy.run_path(str(ROOT / "train_author.py"), run_name="__main__")
    except BaseException as exc:
        append_event({"event": "interrupted", "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        sys.argv = old_argv
    from training_bridge import session
    session.original_save = None
    gc.collect()
    torch.cuda.empty_cache()
    evaluate_best(config, "validation", torch.device("cuda:" + gpu))


def selected_best():
    complete = read(ROOT / "runs/training_complete.json")
    best = read(ROOT / "checkpoints/best.json")
    pid = protocol_id()
    if complete["completed_epoch"] != 200 or complete["protocol_id"] != pid or best["protocol_id"] != pid or sha(ROOT / best["path"]) != best["sha256"]:
        raise ValueError("Complete 200 author-code epochs and retain the selected best first")
    return best


def evaluate_best(config, split, device):
    from network import inference_generator
    from evaluation import evaluate
    best = selected_best()
    if split == "test":
        frozen = read(ROOT / "runs/frozen.json")
        if frozen["best"] != best or frozen["protocol_id"] != protocol_id():
            raise ValueError("Freeze this exact completed run before test evaluation")
    precision(config)
    payload = torch.load(ROOT / best["path"], map_location="cpu", weights_only=False)
    model = inference_generator(payload, device)
    del payload
    output = ROOT / f"results/{split}/epoch_{best['epoch']:03d}"
    summary = evaluate(model, Inputs(read(ROOT / "binding.json")), split, config, device, best["epoch"], output,
                       True, protocol_id(), best["sha256"], frozen=split == "test")
    summary["author_selection"] = best
    summary["formal_metrics_differ_from_random_crop_training_validation"] = True
    write(output / "summary.json", summary)
    print(json.dumps({"report": str(output / "index.html"), "metrics": summary["patient_mean"]}, indent=2))


def freeze():
    best = selected_best()
    report = read(ROOT / f"results/validation/epoch_{best['epoch']:03d}/summary.json")
    if report["checkpoint_sha256"] != best["sha256"] or report["patients"] != 10 or not report["complete_float_predictions_saved"]:
        raise ValueError("Finish the selected best's final validation report first")
    record = {"best": best, "protocol_id": protocol_id(), "test_membership_fixed": True,
              "selection": "author_random_crop_MAE_minus1_plus1", "no_test_arrays_opened": True}
    path = ROOT / "runs/frozen.json"
    if path.exists() and read(path) != record:
        raise ValueError("Existing freeze differs")
    write(path, record)
    print("Frozen. Test arrays have not been opened. Run the separate test action when ready.")


if __name__ == "__main__":
    os.chdir(ROOT)  # Released teacher paths are relative to the experiment directory.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "train", "status", "evaluate", "freeze", "test"))
    parser.add_argument("--parent", type=Path, default=ROOT.parent / "exp011_20261026_fp32_prior_residualfm_100e100e")
    for name in ("labels-root", "data-root", "cache", "manifest"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--gpu", default="0")
    args = parser.parse_args()
    config = read(ROOT / "experiment.json")
    if args.action == "prepare":
        prepare(args.parent, args.labels_root, args.cache, args.manifest, args.data_root)
        write(ROOT / "verification/preparation.json", {"status": "passed", "protocol_id": protocol_id(), "training_started": False, "test_arrays_opened": False})
        print("Prepared existing cache and 150 train/validation masks. No training started.")
    elif args.action == "status":
        print(json.dumps(read(ROOT / "runs/status.json") if (ROOT / "runs/status.json").exists() else {"state": "not_started"}, indent=2))
    elif args.action == "freeze":
        freeze()
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; no silent CPU or precision fallback")
        torch.cuda.set_device(int(args.gpu))
        with exclusive_run():
            if args.action == "check":
                check(config, args.gpu)
            elif args.action == "train":
                train(config, args.gpu)
            else:
                evaluate_best(config, "test" if args.action == "test" else "validation", torch.device("cuda:" + args.gpu))

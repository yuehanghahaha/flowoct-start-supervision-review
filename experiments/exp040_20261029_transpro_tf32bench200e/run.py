"""Prepare, accept on A100, train 200 epochs, freeze, and evaluate independently."""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from experiment import ROOT, read, write, sha, identity, precision, runtime, checkpoint, rng_state, restore_rng, append_event
from dataset import Inputs, prepare, crop
from network import TransPro, inference_generator
from evaluation import evaluate
from retention import storage_budget, store_best, restore_best


def protocol_id():
    return hashlib.sha256((identity() + sha(ROOT / "binding.json")).encode()).hexdigest()


def settings():
    config = read(ROOT / "experiment.json")
    if config["epochs"] != 200 or config["batch_size"] != 1 or config["crop_yzx"] != [256, 256, 256]:
        raise ValueError("Formal budget/geometry changed; create a new numbered experiment")
    if config["loss_weights"] != {"volume": 10.0, "projection": 10.0, "vpg": 5.0, "hcg": 5.0}:
        raise ValueError("This experiment fixes the four coefficients to the paper values")
    if config["optimizer"] != "Adam" or config["gan_objective"] != "paper_logarithmic_minimax" or config["vpg_representation"] != "raw_logits":
        raise ValueError("Loss/optimizer options differ from this implementation; use a new experiment")
    return config


def learning_rate(epoch, config):
    return config["learning_rate"] * (1 if epoch <= 100 else (201 - epoch) / 100)


@contextmanager
def exclusive_run():
    if os.name != "posix":
        raise RuntimeError("Formal training runs on the Linux GPU server")
    import fcntl
    (ROOT / "runs").mkdir(exist_ok=True)
    with (ROOT / "runs/train.lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid()))
        lock.flush()
        yield


def save_unique(path, payload):
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        if existing["protocol_id"] != payload["protocol_id"] or existing["epoch"] != payload["epoch"]:
            raise ValueError("Existing formal checkpoint belongs to another run")
        for name, state in payload["models"].items():
            if set(state) != set(existing["models"][name]) or any(not torch.equal(value, existing["models"][name][key]) for key, value in state.items()):
                raise ValueError("Existing formal checkpoint differs; do not overwrite it")
        return
    checkpoint(path, payload)


def check(config, device):
    precision(config, config["seed"])
    Inputs(read(ROOT / "binding.json"))
    pid = protocol_id()
    result = {"protocol_id": pid, "runtime": runtime(device), "synthetic_only": True,
              "formal_training_started": False, "full_crop": [256, 256, 256], "width": 64}
    try:
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        model = TransPro(config, device)
        optimizer_g, optimizer_d = model.optimizers(config)
        source = 2 * torch.rand((1, 1, 256, 256, 256), device=device, dtype=torch.float32) - 1
        target = 2 * torch.rand_like(source) - 1
        result["losses"] = model.step(source, target, optimizer_g, optimizer_d, config["loss_weights"])
        for optimizer in (optimizer_g, optimizer_d):
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor) and value.is_floating_point() and value.dtype != torch.float32:
                        raise ValueError("Non-FP32 optimizer state: " + key)
        torch.cuda.synchronize(device)
        trained_bytes = sum(value.numel() * value.element_size() for name in ("G", "D3", "D2") for value in getattr(model, name).state_dict().values())
        generator_bytes = sum(value.numel() * value.element_size() for value in model.G.state_dict().values())
        budget = storage_budget(trained_bytes, generator_bytes, config["epochs"], config["checkpoint_every"])
        required_storage = budget["required_bytes"]
        available_storage = shutil.disk_usage(ROOT).free
        result.update(status="passed", seconds=time.perf_counter() - started,
                      max_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      max_reserved_bytes=torch.cuda.max_memory_reserved(device),
                      filesystem_free_bytes=available_storage,
                      conservative_future_storage_bytes=required_storage,
                      storage_policy="paper-selected best, latest resume, final full validation/test predictions",
                      storage_budget=budget,
                      gpu_step_passed=True)
        if available_storage < required_storage:
            raise RuntimeError(f"Measured free storage {available_storage / 1024**3:.1f} GiB is below the conservative retained-output budget {required_storage / 1024**3:.1f} GiB. Use a sufficiently large experiment filesystem; do not silently omit outputs.")
    except Exception as exc:
        result.update(status="failed", error=repr(exc), traceback=traceback.format_exc(),
                      max_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      max_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write(ROOT / "verification/a100_check.json", result)
        raise
    write(ROOT / "verification/a100_check.json", result)
    print(json.dumps(result, indent=2), flush=True)
    print("A100 acceptance passed. Formal training has NOT started.", flush=True)


def train(config, device):
    accepted = read(ROOT / "verification/a100_check.json")
    pid = protocol_id()
    if accepted["status"] != "passed" or accepted["protocol_id"] != pid:
        raise ValueError("Run the full-size A100 check for this exact source/data configuration first")
    precision(config, config["seed"])
    inputs = Inputs(read(ROOT / "binding.json"))
    rows = inputs.split("train")
    if len(rows) != 140:
        raise ValueError("Expected 140 fixed training patients")
    model = TransPro(config, device)
    optimizer_g, optimizer_d = model.optimizers(config)
    latest = ROOT / "checkpoints/latest.pt"
    completed, best = 0, {"MAE": float("inf"), "epoch": None}
    best_record = None
    history = []
    if latest.exists():
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        if payload["protocol_id"] != pid:
            raise ValueError("Resume source/data/metric protocol changed")
        for name, state in payload["models"].items():
            getattr(model, name).load_state_dict(state)
        optimizer_g.load_state_dict(payload["optimizer_g"])
        optimizer_d.load_state_dict(payload["optimizer_d"])
        restore_rng(payload["rng"])
        completed, best, history = payload["epoch"], payload["best"], payload["history"]
        best_record = payload["best_record"]
        restore_best(best_record)
        del payload
        gc.collect()
    partial_updates = 0
    epoch = completed
    try:
        for epoch in range(completed + 1, config["epochs"] + 1):
            started = time.perf_counter()
            rate = learning_rate(epoch, config)
            for optimizer in (optimizer_g, optimizer_d):
                for group in optimizer.param_groups:
                    group["lr"] = rate
            order = np.random.default_rng(np.random.SeedSequence([config["seed"], epoch, 1])).permutation(len(rows))
            partial_updates, totals = 0, {}
            for index, row_index in enumerate(order):
                row = rows[int(row_index)]
                generator = np.random.default_rng(np.random.SeedSequence([config["seed"], epoch, index, 2]))
                y, x = (int(value) for value in generator.integers(0, 49, size=2))
                source, target = inputs.pair(row)
                a, b = [torch.from_numpy(crop(array, y, x)).to(device=device, dtype=torch.float32) for array in (source, target)]
                losses = model.step(a, b, optimizer_g, optimizer_d, config["loss_weights"])
                partial_updates += 1
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0) + value
                append_event({"event": "update", "epoch": epoch, "patient": row["case_id"], "crop_yx": [y, x], "step": partial_updates, "losses": losses})
                if (index + 1) % 10 == 0:
                    print(f"epoch {epoch}/200 | patient {index + 1}/140 | G={losses['G']:.5f} D={losses['D']:.5f}", flush=True)
                del source, target, a, b
            summary = evaluate(model.G, inputs, "validation", config, device, epoch,
                               ROOT / f"metrics/validation/epoch_{epoch:03d}", False, pid)
            mae = summary["patient_mean"]["MAE"]
            if not np.isfinite(mae):
                raise FloatingPointError("Validation MAE is not finite")
            if mae < best["MAE"]:
                best = {"MAE": mae, "epoch": epoch}
                state = {"G": {key: value.detach().cpu() for key, value in model.G.state_dict().items()}}
                best_record = store_best({"epoch": epoch, "models": state, "best": best, "config": config, "protocol_id": pid})
                del state
            entry = {"epoch": epoch, "updates": epoch * len(rows), "learning_rate": rate,
                     "losses": {key: value / len(rows) for key, value in totals.items()},
                     "validation": summary["patient_mean"], "best": best.copy(), "seconds": time.perf_counter() - started}
            history.append(entry)
            states = model.trained_state()
            if config["checkpoint_every"] and epoch % config["checkpoint_every"] == 0:
                save_unique(ROOT / f"checkpoints/epoch_{epoch:03d}.pt", {"epoch": epoch, "models": states, "config": config, "protocol_id": pid})
            checkpoint(latest, {"epoch": epoch, "models": states, "optimizer_g": optimizer_g.state_dict(),
                                "optimizer_d": optimizer_d.state_dict(), "rng": rng_state(), "best": best,
                                "history": history, "config": config, "protocol_id": pid, "best_record": best_record})
            write(ROOT / "runs/history.json", history)
            write(ROOT / "runs/status.json", {"state": "training" if epoch < 200 else "complete", "epoch": epoch,
                                               "epochs": 200, "updates": epoch * 140, "best": best, "protocol_id": pid})
            append_event({"event": "epoch_complete", **entry})
            completed = epoch
            del states
            gc.collect()
        write(ROOT / "runs/training_complete.json", {"epochs": 200, "updates": 28000, "best": best, "protocol_id": pid})
    except BaseException as exc:
        append_event({"event": "training_interrupted", "epoch": epoch, "completed_epoch": completed,
                      "partial_updates": partial_updates, "error": repr(exc), "traceback": traceback.format_exc()})
        write(ROOT / "runs/status.json", {"state": "interrupted", "completed_epoch": completed, "partial_epoch": epoch,
                                           "partial_updates": partial_updates, "best": best, "error": repr(exc)})
        # Logs retain the failure evidence; automatic resume uses the last completed epoch.
        raise
    # No more optimizer updates: generate full validation outputs only for the final selected best.
    del model, optimizer_g, optimizer_d
    gc.collect()
    torch.cuda.empty_cache()
    evaluate_best(config, device, "validation")


def freeze():
    complete, best = read(ROOT / "runs/training_complete.json"), read(ROOT / "checkpoints/best.json")
    pid = protocol_id()
    if complete["epochs"] != 200 or complete["protocol_id"] != pid or best["protocol_id"] != pid or sha(ROOT / best["path"]) != best["sha256"]:
        raise ValueError("Training must be complete and the selected checkpoint unchanged")
    validation = read(ROOT / f"results/validation/epoch_{best['epoch']:03d}/summary.json")
    if validation["checkpoint_sha256"] != best["sha256"] or not validation["complete_float_predictions_saved"] or validation["patients"] != 10:
        raise ValueError("Finish the final best validation report before freezing")
    record = {"protocol_id": pid, "best": best, "completed_epochs": 200, "test_membership_fixed": True,
              "no_test_arrays_opened": True, "selection": "minimum_validation_MAE"}
    existing = ROOT / "runs/frozen.json"
    if existing.exists() and read(existing) != record:
        raise ValueError("An existing freeze record differs")
    write(existing, record)
    print("Frozen. No test arrays have been opened. Run the separate test command when ready.")


def evaluate_best(config, device, split):
    pid = protocol_id()
    completed = read(ROOT / "runs/training_complete.json")
    if completed["epochs"] != 200 or completed["protocol_id"] != pid:
        raise ValueError("This retention policy saves complete predictions after the full 200 epochs")
    best = read(ROOT / "checkpoints/best.json")
    if best["protocol_id"] != pid or sha(ROOT / best["path"]) != best["sha256"]:
        raise ValueError("Selected checkpoint/configuration changed")
    if split == "test":
        frozen = read(ROOT / "runs/frozen.json")
        if frozen["protocol_id"] != pid or frozen["best"] != best:
            raise ValueError("Freeze record differs")
    precision(config, config["seed"])
    payload = torch.load(ROOT / best["path"], map_location="cpu", weights_only=False)
    generator = inference_generator(payload["models"]["G"], device)
    del payload
    output = ROOT / f"results/{split}/epoch_{best['epoch']:03d}"
    result = evaluate(generator, Inputs(read(ROOT / "binding.json")), split, config, device, best["epoch"], output,
                      True, pid, best["sha256"], frozen=split == "test")
    if split == "validation" and abs(result["patient_mean"]["MAE"] - best["MAE"]) > 1e-6:
        raise ValueError("Saved final predictions differ from the checkpoint-selection score")
    print(json.dumps({"report": str(output / "index.html"), "metrics": result["patient_mean"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "check", "train", "status", "evaluate", "freeze", "test"))
    parser.add_argument("--parent", type=Path, default=ROOT.parent / "exp011_20261026_fp32_prior_residualfm_100e100e")
    parser.add_argument("--labels-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = settings()
    if args.action == "prepare":
        result = prepare(args.parent, args.labels_root, args.cache, args.manifest, args.data_root)
        write(ROOT / "verification/preparation.json", {"status": "passed", "protocol_id": protocol_id(),
                                                       "cache": result["cache"], "labels_root": result["labels_root"],
                                                       "train_patients": 140, "validation_patients": 10,
                                                       "test_arrays_opened": False, "training_started": False})
        print("Prepared existing cache and 150 training/validation masks. No training started.")
    elif args.action == "status":
        print(json.dumps(read(ROOT / "runs/status.json") if (ROOT / "runs/status.json").exists() else {"state": "not_started"}, indent=2))
    elif args.action == "freeze":
        freeze()
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("A working CUDA GPU is required for formal training/evaluation")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        with exclusive_run():
            if args.action == "check":
                check(config, device)
            elif args.action == "train":
                train(config, device)
            else:
                evaluate_best(config, device, "test" if args.action == "test" else "validation")

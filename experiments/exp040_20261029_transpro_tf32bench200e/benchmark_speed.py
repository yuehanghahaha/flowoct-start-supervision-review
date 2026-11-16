"""Synthetic same-GPU TransPro speed comparison; no patient data or formal training."""
import argparse
import csv
from datetime import datetime, timezone
import os
import statistics
import subprocess
import sys
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # Match exp028 for every profile.
import torch

from experiment import ROOT, read, write, identity, precision, runtime
from network import TransPro


PROFILES = {
    "baseline": {"cudnn_tf32": False, "benchmark": False, "deterministic": True},
    "tf32_only": {"cudnn_tf32": True, "benchmark": False, "deterministic": True},
    "algorithms_only": {"cudnn_tf32": False, "benchmark": True, "deterministic": False},
    "combined": {"cudnn_tf32": True, "benchmark": True, "deterministic": False},
}


def apply_profile(name):
    profile = PROFILES[name]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = profile["cudnn_tf32"]
    torch.backends.cudnn.benchmark = profile["benchmark"]
    torch.backends.cudnn.deterministic = profile["deterministic"]


def worker(args, output):
    path = output / (args.worker + ".json")
    report = {"profile": args.worker, "requested_flags": PROFILES[args.worker],
              "synthetic_only": True, "formal_training_started": False, "patient_arrays_opened": False,
              "full_crop": [256, 256, 256], "width": 64, "batch_size": 1,
              "seed": 2026, "warmup_steps": args.warmup, "measured_steps": args.steps,
              "includes": "Exact exp028 G/D/teacher model.step, optimizer updates and per-parameter checks; no loader/validation/checkpoint I/O"}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("A CUDA GPU is required; no CPU or smaller-model fallback")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        config = read(ROOT / "experiment.json")
        precision(config, 2026)
        apply_profile(args.worker)
        report["runtime"] = runtime(device)
        started = time.perf_counter()
        model = TransPro(config, device)
        optimizer_g, optimizer_d = model.optimizers(config)
        source = torch.rand((1, 1, 256, 256, 256), device=device, dtype=torch.float32) * 2 - 1
        target = torch.rand_like(source) * 2 - 1
        torch.cuda.synchronize(device)
        report["model_and_input_setup_seconds"] = time.perf_counter() - started
        report["input_sums"] = {"source":float(source.double().sum()), "target":float(target.double().sum())}
        report["warmup_seconds"] = []
        for index in range(args.warmup):
            torch.cuda.synchronize(device)
            before = time.perf_counter()
            model.step(source, target, optimizer_g, optimizer_d, config["loss_weights"])
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - before
            report["warmup_seconds"].append(elapsed)
            print(f"{args.worker} warmup {index + 1}/{args.warmup}: {elapsed:.3f}s", flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        times = []
        for index in range(args.steps):
            torch.cuda.synchronize(device)
            before = time.perf_counter()
            losses = model.step(source, target, optimizer_g, optimizer_d, config["loss_weights"])
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - before
            times.append(elapsed)
            print(f"{args.worker} measured {index + 1}/{args.steps}: {elapsed:.3f}s", flush=True)
        report.update(status="passed", step_seconds=times, mean_seconds=statistics.mean(times),
                      median_seconds=statistics.median(times), min_seconds=min(times), max_seconds=max(times),
                      last_synthetic_losses=losses,
                      max_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      max_reserved_bytes=torch.cuda.max_memory_reserved(device))
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        write(path, report)
        raise
    write(path, report)
    return report


def summarize(reports):
    baseline = next((row for row in reports if row.get("profile") == "baseline" and row.get("status") == "passed"), None)
    result = []
    for row in reports:
        result.append({"profile":row["profile"], "status":row["status"],
                       "mean_seconds":row.get("mean_seconds"), "median_seconds":row.get("median_seconds"),
                       "speedup_vs_baseline":baseline["mean_seconds"] / row["mean_seconds"] if baseline and row.get("mean_seconds") else None,
                       "max_allocated_GiB":row["max_allocated_bytes"] / 1024**3 if "max_allocated_bytes" in row else None})
    return result


def main(args):
    root = ROOT / "verification/speed_benchmarks"
    root.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + str(time.time_ns())
    output = root / run_name
    output.mkdir()
    write(output / "plan.json", {"source_configuration_identity":identity(), "profiles":args.profiles,
          "profile_settings":PROFILES, "warmup":args.warmup, "steps":args.steps, "device":args.device,
          "formal_training_started":False, "no_patient_arrays":True,
          "interpretation":"GPU step-only comparison. Not clinical accuracy, epoch duration, or author-equivalence evidence."})
    reports = []
    for name in args.profiles:
        command = [sys.executable, "-B", "-u", str(ROOT / "benchmark_speed.py"), "--worker", name,
                   "--output", str(output), "--device", args.device, "--warmup", str(args.warmup), "--steps", str(args.steps)]
        print(f"Starting fresh process: {name}", flush=True)
        process = subprocess.run(command, cwd=ROOT)
        path = output / (name + ".json")
        report = read(path) if path.exists() else {"profile":name,"status":"failed","error":f"Worker exited {process.returncode} without a result"}
        reports.append(report)
    rows = summarize(reports)
    write(output / "summary.json", {"rows":rows,"raw_profiles":reports,"formal_training_started":False,
          "timing_scope":"Warm steps of exp028 model.step only, including its gradient checks; no validation or disk I/O"})
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = ("profile","status","mean_seconds","median_seconds","speedup_vs_baseline","max_allocated_GiB")
        csv_writer = csv.DictWriter(stream, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    for row in rows:
        if row["status"] == "passed":
            speed = "n/a" if row["speedup_vs_baseline"] is None else f"{row['speedup_vs_baseline']:.2f}x"
            print(f"{row['profile']}: {row['mean_seconds']:.3f}s/update, speedup={speed}", flush=True)
        else:
            print(f"{row['profile']}: FAILED; inspect its JSON error", flush=True)
    print("Results:", output, flush=True)
    return 0 if all(row["status"] == "passed" for row in rows) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", nargs="+", choices=tuple(PROFILES), default=list(PROFILES))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker", choices=tuple(PROFILES), help=argparse.SUPPRESS)
    parser.add_argument("--output", type=str, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup < 1 or args.steps < 1:
        parser.error("Use at least one warmup and one measured step")
    if args.worker:
        if not args.output:
            parser.error("Worker mode requires --output")
        from pathlib import Path
        output = Path(args.output).resolve()
        if not output.is_relative_to(ROOT / "verification/speed_benchmarks"):
            parser.error("Worker output must stay in this experiment's speed benchmark directory")
        worker(args, output)
    else:
        raise SystemExit(main(args))

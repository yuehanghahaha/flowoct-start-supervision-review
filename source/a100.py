"""Control the A100 tmux job from lyh_tmux using only the standard library."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
SESSION = "exp172_fullvolume_rms_mamba_bf16"
DEFAULT_DATASET = Path("/workspace/flowoct_work/data/OCTA500_v1")
OLDER_MANIFEST = Path("/workspace/experience/exp004_20261025_a100_prior_direct_residualfm_100e100e/data/manifest.csv")
PAUSE_EXIT = 75


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def optional_json(path):
    return read_json(path) if path.is_file() else None


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            value.update(block)
    return value.hexdigest()


def verify_sources(root):
    for line in (root / "sources.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if digest(root / relative) != expected:
            raise ValueError(f"Uploaded file checksum differs: {relative}")


def tmux(*args, **options):
    return subprocess.run(["tmux", *args], **options)


def session_panes():
    if shutil.which("tmux") is None:
        return []
    result = tmux("list-panes", "-s", "-t", f"={SESSION}", "-F", "#{pane_id} #{pane_dead}",
                  capture_output=True, text=True)
    if result.returncode:
        return []
    return [dict(pane=line.split()[0], dead=line.split()[1] == "1")
            for line in result.stdout.splitlines() if line.strip()]


def session_live():
    return any(not pane["dead"] for pane in session_panes())


@contextmanager
def launch_lock(root):
    import fcntl

    path = root / "runs/a100/controller.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def data_paths(args):
    manifest = Path(args.manifest) if args.manifest else DEFAULT_DATASET / "split_3M_z256.csv"
    if not args.manifest and not manifest.is_file() and OLDER_MANIFEST.is_file():
        manifest = OLDER_MANIFEST
    layers = Path(args.labels_root) if args.labels_root else DEFAULT_DATASET / "labels_native"
    data = Path(args.data_root) if args.data_root else DEFAULT_DATASET / "z256/3M"
    if not manifest.is_file() or not (layers / "GT_Layers").is_dir() or not data.is_dir():
        raise FileNotFoundError(f"Check the A100 mount. Manifest={manifest}; pairs={data}; layers={layers}")
    return manifest, layers, data


def run_python(*args, capture=False):
    return subprocess.run([sys.executable, "-B", "-u", *map(str, args)], cwd=ROOT, check=True,
                          text=True, capture_output=capture)


def probe():
    import causal_conv1d
    import mamba_ssm
    import torch
    from runtime import configure, load_config, source_contract

    runtime = configure(load_config(), torch.device("cuda:0"))
    if "A100" not in runtime["gpu"] or "MIG" in runtime["gpu"]:
        raise RuntimeError(f"Expected an allocated full A100, found {runtime['gpu']}.")
    from run import signature_for
    signature = signature_for()
    print(json.dumps(dict(runtime=runtime, signature=signature)))


def stage(root, action, phase, state="running", **extra):
    atomic_json(root / "runs/a100/pipeline.json", dict(action=action, phase=phase, state=state, at=stamp(), **extra))


def check_needed(probe_result, existing, force=False):
    return (force or not existing or existing.get("state") != "passed"
            or existing.get("signature") != probe_result["signature"]
            or existing.get("runtime") != probe_result["runtime"])


def worker(args):
    action = args.action
    phase = "source_check"

    def enter(value):
        nonlocal phase
        phase = value
        if (ROOT / "runs/pause.request").is_file():
            raise subprocess.CalledProcessError(PAUSE_EXIT, phase)
        stage(ROOT, action, phase)
        print(f"[{stamp()}] {phase}", flush=True)

    try:
        enter("source_check")
        verify_sources(ROOT)
        ready = optional_json(ROOT / "analysis_return/test_complete.json")
        if ready and action != "check":
            enter("verify_return")
            run_python("return_results.py", "verify")
            stage(ROOT, action, "analysis_return", "complete", directory=str(ROOT / "analysis_return"))
            return 0
        if action == "test" and not (ROOT / "runs/training_complete.json").is_file():
            raise RuntimeError("Formal test requires the completed 200-epoch training receipt.")
        receipt = ROOT / "runs/deployment.json"
        if args.manifest or args.labels_root or args.data_root or not receipt.is_file():
            enter("binding")
            manifest, labels, data = data_paths(args)
            run_python("run.py", "bind", "--manifest", manifest, "--labels-root", labels, "--data-root", data)
        enter("cpu_correctness")
        run_python("verify.py")
        enter("training_only_statistics")
        run_python("run.py", "stats")
        enter("a100_probe")
        current = json.loads(run_python("a100.py", "_probe", capture=True).stdout)
        print(f"Allocated GPU: {current['runtime']['gpu']}", flush=True)
        existing = optional_json(ROOT / "verification/full_gpu.json")
        if check_needed(current, existing, force=action == "check"):
            enter("gpu_check")
            run_python("run.py", "check")
        if action == "check":
            stage(ROOT, action, "gpu_check", "passed")
            return 0
        if action != "test" and not (ROOT / "runs/training_complete.json").is_file():
            enter("training")
            run_python("run.py", "train")
        enter("test")
        run_python("run.py", "test")
        enter("post_selection_diagnostics")
        run_python("run.py", "diagnostics")
        enter("export")
        run_python("return_results.py", "collect")
        enter("verify_return")
        run_python("return_results.py", "verify")
        stage(ROOT, action, "analysis_return", "complete", directory=str(ROOT / "analysis_return"))
        return 0
    except subprocess.CalledProcessError as error:
        state = "paused" if error.returncode == PAUSE_EXIT else "failed"
        stage(ROOT, action, phase, state, exit_code=error.returncode)
        if error.stderr:
            print(error.stderr, file=sys.stderr, flush=True)
        print(f"{state}: {phase} (exit {error.returncode})", file=sys.stderr, flush=True)
        return error.returncode
    except Exception as error:
        stage(ROOT, action, phase, "failed", error_type=type(error).__name__, error=str(error))
        raise



def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('_worker', '_probe'):
        parser = argparse.ArgumentParser()
        parser.add_argument('command', choices=('_worker', '_probe'))
        parser.add_argument('action', nargs='?', choices=('start', 'resume', 'check', 'test'))
        for name in ('manifest', 'labels-root', 'data-root'):
            parser.add_argument('--' + name)
        args = parser.parse_args()
        if args.command == '_probe':
            probe()
            return 0
        return worker(args)
    controller = ROOT / 'runs/slot2/control.py'
    os.execv('/workspace/.conda/envs/lyh_flowoct/bin/python',
             ['/workspace/.conda/envs/lyh_flowoct/bin/python', '-B', str(controller), *sys.argv[1:]])


if __name__ == '__main__':
    raise SystemExit(main())

"""Run the delivered experiment in slot2 with the existing lyh_flowoct environment."""
import argparse
from datetime import datetime
import getpass
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/workspace")
PYTHON = WORKSPACE / ".conda/envs/lyh_flowoct/bin/python"
TMUX = WORKSPACE / ".conda/envs/lyh_tmux/bin/tmux"


def checked(path):
    resolved = path.resolve()
    if resolved != WORKSPACE and WORKSPACE not in resolved.parents:
        raise ValueError(f"Path leaves the confirmed user workspace: {path} -> {resolved}")
    return resolved


def configure():
    if getpass.getuser() != "HPC_USER" or Path.home() != WORKSPACE:
        raise RuntimeError("Enter the allocated slot2 container as HPC_USER first.")
    if socket.gethostname() != "hpc-HPC_USER-slot2":
        raise RuntimeError("This controller belongs to the user's slot2 allocation.")
    checked(ROOT)
    mapping = ROOT / 'verification/server_mapping.json'
    if not mapping.is_file():
        raise RuntimeError('Host/container mapping evidence must be registered before launch.')
    evidence = json.loads(mapping.read_text())
    stat = WORKSPACE.stat()
    if [stat.st_dev, stat.st_ino] != evidence['workspace_inode']:
        raise RuntimeError('The current /workspace does not match the verified persistent user home.')
    checked(PYTHON)
    checked(TMUX)
    cache = ROOT / "runs/cache"
    paths = {
        "TMPDIR": cache / "tmp",
        "TMP": cache / "tmp",
        "TEMP": cache / "tmp",
        "XDG_CACHE_HOME": cache,
        "PIP_CACHE_DIR": cache / "pip",
        "TORCH_HOME": cache / "torch",
        "HF_HOME": cache / "huggingface",
        "TRITON_CACHE_DIR": cache / "triton",
        "CUDA_CACHE_PATH": cache / "cuda",
        "TMUX_TMPDIR": cache / "tmux",
    }
    for name, path in paths.items():
        checked(path).mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    paths["TMUX_TMPDIR"].chmod(0o700)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PATH"] = str(TMUX.parent) + ":" + str(PYTHON.parent) + ":" + os.environ.get("PATH", "")
    os.environ.pop("TMUX", None)
    os.environ.pop("TMUX_PANE", None)
    sys.path.insert(0, str(ROOT))
    os.chdir(ROOT)


def launch(a100, action):
    with a100.launch_lock(ROOT):
        if a100.session_live():
            raise RuntimeError("The experiment session is already active. Use status or attach.")
        a100.verify_sources(ROOT)
        run_requested = ROOT / "runs/a100/run_requested.json"
        latest = ROOT / "checkpoints/latest.pt"
        if action == "start" and (run_requested.exists() or latest.exists()):
            raise RuntimeError("This experiment has saved state. Use resume.")
        if a100.session_panes():
            previous = a100.optional_json(ROOT / "runs/a100/last_launch.json")
            if not previous or previous["session"] != a100.SESSION:
                raise RuntimeError("Review the existing session ownership first.")
            a100.tmux("kill-session", "-t", f"={a100.SESSION}", check=True)
        log = ROOT / "runs/a100/logs" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_slot2.log")
        checked(log).parent.mkdir(parents=True, exist_ok=True)
        command = [str(PYTHON), "-B", "-u", "a100.py", "_worker", action]
        shell = "set -eo pipefail\n" + shlex.join(command) + " 2>&1 | tee -a " + shlex.quote(str(log))
        pane_command = shlex.join(["bash", "-c", shell])
        (ROOT / "runs/pause.request").unlink(missing_ok=True)
        a100.atomic_json(ROOT / "runs/a100/last_launch.json",
                         dict(session=a100.SESSION, action=action, slot="slot2", python=str(PYTHON),
                              log=str(log), at=a100.stamp(), controller=str(Path(__file__).resolve())))
        a100.tmux("new-session", "-d", "-s", a100.SESSION, "-c", str(ROOT), pane_command, check=True)
        if action in ("start", "resume"):
            a100.atomic_json(run_requested, dict(action=action, at=a100.stamp(), slot="slot2"))
    print(json.dumps(dict(session=a100.SESSION, slot="slot2", action=action, log=str(log)), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "resume", "pause", "status", "attach", "check", "test", "log"))
    args = parser.parse_args()
    configure()
    import a100

    if args.action in ("start", "resume", "check", "test"):
        launch(a100, args.action)
    elif args.action == "pause":
        with a100.launch_lock(ROOT):
            if not a100.session_live():
                raise RuntimeError("There is no active experiment session. Use status.")
            (ROOT / "runs/pause.request").touch()
        print("Pause requested at the next saved update or patient boundary.")
    elif args.action == "attach":
        a100.tmux("attach-session", "-t", f"={a100.SESSION}", check=True)
    elif args.action == "log":
        record = a100.optional_json(ROOT / "runs/a100/last_launch.json")
        if record:
            log = checked(Path(record["log"]))
            subprocess.run(["tail", "-n", "45", str(log)], check=True)
    else:
        print(json.dumps(dict(slot="slot2", container=socket.gethostname(), session=a100.SESSION,
                              tmux_live=a100.session_live(),
                              pause_requested=(ROOT / "runs/pause.request").is_file(),
                              pipeline=a100.optional_json(ROOT / "runs/a100/pipeline.json"),
                              progress=a100.optional_json(ROOT / "runs/status.json"),
                              launch=a100.optional_json(ROOT / "runs/a100/last_launch.json"),
                              ready_to_download=(ROOT / "analysis_return/test_complete.json").is_file()),
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Start the four independent formal experiments in separate tmux sessions."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def active(session):
    return shutil.which('tmux') is not None and subprocess.run(
        ['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['start', 'status'])
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--labels-root', type=Path)
    parser.add_argument('--only', type=int, nargs='+', choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    args = parser.parse_args()
    names = read(ROOT/'suite.json')['experiments']
    if args.action == 'start' and (os.name != 'posix' or not shutil.which('tmux')):
        raise RuntimeError('Run on the Linux A100 host with tmux installed')
    scheduled = []
    for number in args.only:
        folder = ROOT.parent/names[number-1]
        config = read(folder/'experiment.json')
        session = f'fm3d_{folder.name[:6]}'
        status_path = folder/'runs/train/status.json'
        status = read(status_path) if status_path.exists() else {'state': 'not_started'}
        if args.action == 'status':
            print(json.dumps(dict(experiment=folder.name, session=session, tmux_running=active(session), **status), ensure_ascii=False))
            continue
        if active(session) or status.get('state') == 'complete':
            print(folder.name, 'already running or complete; left unchanged'); continue
        if (folder/'verification/delivery_sources.json').exists():
            import hashlib
            for rel, digest in read(folder/'verification/delivery_sources.json').items():
                if hashlib.sha256((folder/rel).read_bytes()).hexdigest() != digest:
                    raise ValueError(f'Delivered source changed: {folder.name}/{rel}')
        bind = [sys.executable, '-B', str(folder/'run.py'), 'bind']
        for key in ['data_root', 'manifest', 'labels_root']:
            value = getattr(args, key)
            if value is not None: bind += ['--'+key.replace('_', '-'), str(value.resolve())]
        # Only resolves files and fixed split membership; never performs a model check.
        subprocess.run(bind, cwd=folder, check=True)
        mode = 'resume' if any((folder/'checkpoints'/name).exists() for name in ['latest.pt', 'initial.pt']) else 'train'
        command = [sys.executable, '-B', '-u', str(folder/'run.py'), mode]
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log = folder/'logs'/f'formal_{stamp}.log'
        variables = {'PYTHONUNBUFFERED': '1', 'NVIDIA_TF32_OVERRIDE': '0', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE': '0'}
        for key in ['PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'CONDA_PREFIX']:
            if key in os.environ: variables[key] = os.environ[key]
        shell_command = shlex.join(['env']+[f'{k}={v}' for k, v in variables.items()]+command)
        shell_command += ' >> '+shlex.quote(str(log))+' 2>&1'
        scheduled.append((folder, session, shell_command, log, config))
    # Resolve every selected job before starting any of the new jobs.
    for folder, session, command, log, config in scheduled:
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(folder), command], check=True)
        record = dict(experiment=folder.name, session=session, started_at=datetime.now().astimezone().isoformat(),
            python=sys.executable, log=str(log), epochs=config['epochs'], independent_process=True,
            inherited_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
            automatic_precision_or_patch_reduction=False)
        (folder/'logs/launch.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        (folder/'logs'/f'launch_{stamp}.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(record))


if __name__ == '__main__':
    main()

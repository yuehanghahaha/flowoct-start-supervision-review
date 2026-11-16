"""Prepare, start and inspect the two independently retained A100 experiments."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import common as c


def directories():
    cfg = c.read(c.ROOT/'experiment.json')
    paths = [c.ROOT.parent/name for name in cfg['suite_experiments']]
    if not all(path.is_dir() for path in paths):
        raise FileNotFoundError('Upload both experiment folders next to each other')
    return paths


def python(path, *arguments):
    subprocess.run([sys.executable, '-B', '-u', *arguments], cwd=path, check=True)


def tmux_executable(environment):
    conda = os.environ.get('CONDA_EXE') or shutil.which('conda')
    if not conda:
        raise RuntimeError('Activate the training conda environment before starting tmux')
    result = subprocess.run([conda, 'run', '--no-capture-output', '-n', environment, 'which', 'tmux'],
                            capture_output=True, text=True, check=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip().startswith('/')]
    if len(lines) != 1 or not Path(lines[0]).is_file():
        raise FileNotFoundError('Install tmux into the lyh_tmux conda environment')
    return lines[0]


def conda_script():
    binary = os.environ.get('CONDA_EXE')
    candidates = [Path(binary).resolve().parent.parent/'etc/profile.d/conda.sh'] if binary else []
    candidates.extend((Path('/opt/conda/etc/profile.d/conda.sh'),
                       Path.home()/'miniconda3/etc/profile.d/conda.sh'))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError('Source the installed conda.sh and activate the training environment')


def start(paths, environment):
    tmux = tmux_executable(environment)
    init = conda_script()
    if Path(os.environ.get('CONDA_PREFIX', '')).resolve() != Path(sys.prefix).resolve():
        raise RuntimeError('Activate the conda environment corresponding to the current Python')
    for path in paths:
        python(path, 'verify_sources.py')
        cfg = c.read(path/'experiment.json')
        if (path/'runs/train/training_complete.json').exists():
            print(path.name+' has already completed its budget.', flush=True)
            continue
        for check in ('preparation.json', 'preflight.json', 'full_preflight.json'):
            record = c.read(path/'verification'/check)
            if not record['passed'] or any(c.sha(path/name) != expected
                    for name, expected in record['source_contract'].items()):
                raise ValueError('Acceptance did not pass: '+str(path/check))
            if 'versions' in record and record['versions'] != c.versions():
                raise ValueError('Environment changed after acceptance: '+path.name)
        session = cfg['experiment_id'].split('_')[0]+'_layerproj'
        if subprocess.run([tmux, 'has-session', '-t', '='+session], capture_output=True).returncode == 0:
            print(session+' is already running.', flush=True)
            continue
        action = 'resume' if (path/'checkpoints/initial.pt').exists() else 'train'
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        log = path/'logs'/('train_'+stamp+'.log')
        exit_file = log.with_suffix('.exit')
        script = path/'logs'/('tmux_'+stamp+'.sh')
        path.joinpath('logs').mkdir(exist_ok=True)
        command = shlex.join([sys.executable, '-B', '-u', 'run.py', action])
        gpu_environment = {key: os.environ.get(key) for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER',
            'NVIDIA_VISIBLE_DEVICES', 'NVIDIA_TF32_OVERRIDE', 'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')}
        exports = ''.join(('unset '+key+'\n') if value is None else
                          ('export '+key+'='+shlex.quote(value)+'\n') for key, value in gpu_environment.items())
        script.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            +'source '+shlex.quote(str(init))+'\n'
            +'conda activate '+shlex.quote(sys.prefix)+'\n'
            +exports
            +'cd '+shlex.quote(str(path))+'\n'
            +'set +e\n'
            +command+' > '+shlex.quote(str(log))+' 2>&1\n'
            +'code=$?\nprintf "%s\\n" "$code" > '+shlex.quote(str(exit_file))+'\nexit "$code"\n', encoding='utf-8')
        subprocess.run([tmux, 'new-session', '-d', '-s', session, '-c', str(path),
                        shlex.join(['bash', str(script)])], check=True)
        c.write(path/'verification/latest_launch.json', dict(session=session, action=action,
                python=sys.executable, conda_prefix=sys.prefix, conda_source=str(init), tmux=tmux,
                gpu_environment=gpu_environment,
                log=str(log), exit_file=str(exit_file), script=str(script),
                session_created=True, training_steps_confirmed=False))
        print(session+' launched. Log '+str(log), flush=True)


def status(paths):
    for path in paths:
        record = path/'runs/train/status.json'
        launch = path/'verification/latest_launch.json'
        result = dict(experiment=path.name, training=c.read(record) if record.exists() else {'state': 'not_started'})
        if launch.exists():
            result['launch'] = c.read(launch)
            exit_file = Path(result['launch']['exit_file'])
            result['exit_code'] = exit_file.read_text().strip() if exit_file.exists() else None
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action', choices=('verify', 'prepare', 'start', 'status', 'report', 'test'))
    cli.add_argument('--tmux-env', default='lyh_tmux')
    for key in ('data-root', 'manifest', 'cache-root', 'labels-root'):
        cli.add_argument('--'+key)
    args = cli.parse_args()
    paths = directories()
    if args.action == 'status':
        status(paths)
    elif args.action == 'start':
        start(paths, args.tmux_env)
    elif args.action == 'report':
        python(paths[0], 'compare.py', '--split', 'validation')
    elif args.action == 'test':
        for path in paths:
            python(path, 'run.py', 'freeze')
            python(path, 'run.py', 'test')
        python(paths[0], 'compare.py', '--split', 'test')
    else:
        arguments = []
        for key in ('data_root', 'manifest', 'cache_root', 'labels_root'):
            value = getattr(args, key)
            if value:
                arguments.extend(('--'+key.replace('_', '-'), value))
        for path in paths:
            python(path, 'verify_sources.py')
            if args.action == 'prepare':
                python(path, 'prepare.py', *arguments)
                python(path, 'checks.py', 'synthetic')
                python(path, 'checks.py', 'real')
        print('Both deliveries verified.' if args.action == 'verify' else
              'Both experiments prepared and accepted. Formal training has not started.', flush=True)


if __name__ == '__main__':
    main()

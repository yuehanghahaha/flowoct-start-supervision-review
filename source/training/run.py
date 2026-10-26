"""Run this frozen experiment: check, train, resume, status, or test."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def load_config():
    return json.loads((ROOT / 'experiment.json').read_text(encoding='utf-8'))


def training_command(config, manifest):
    command = [sys.executable, '-u', '-m', 'fm3d.train_start']
    for key in ('fov', 'axis_order', 'intensity_scale', 'depth', 'base_channels',
                'epochs', 'steps_per_epoch', 'start_fraction', 'lr', 'val_every',
                'save_every', 'ode_steps', 'overlap', 'seed', 'device'):
        command.extend(['--' + key.replace('_', '-'), str(config[key])])
    command.extend(['--manifest', str(manifest), '--output', str(ROOT / config['output']),
                    '--layers-dir', str(ROOT / config['layers_dir']),
                    '--patch', *map(str, config['patch']),
                    '--amp' if config['amp'] else '--no-amp'])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'train', 'resume', 'status', 'test'))
    parser.add_argument('--manifest', type=Path, help='Override data location only; scientific settings stay in experiment.json')
    args = parser.parse_args()
    config = load_config()
    output = ROOT / config['output']
    manifest = (args.manifest or Path(os.environ.get('FM3D_MANIFEST', config['manifest']))).resolve()
    if args.action == 'status':
        status = output / 'status.json'
        print(status.read_text(encoding='utf-8') if status.exists() else
              json.dumps(dict(state='not_started', epochs=config['epochs'], output=str(output)), indent=2))
        return
    if args.action == 'check':
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
        command = [sys.executable, '-u', 'verify.py', '--manifest', str(manifest),
                   '--output', str(ROOT / 'verification' / ('preflight_' + stamp))]
    elif args.action in ('train', 'resume'):
        command = training_command(config, manifest)
        if args.action == 'resume':
            command.append('--resume')
    else:
        if not (output / 'status.json').exists():
            raise FileNotFoundError('No completed training run')
        status = json.loads((output / 'status.json').read_text(encoding='utf-8'))
        if status['state'] != 'complete':
            raise RuntimeError('Finish the frozen training budget before final test evaluation')
        command = [sys.executable, '-u', '-m', 'fm3d.evaluate', '--checkpoint', str(output / 'best.pt'),
                   '--split', 'test', '--output', str(ROOT / 'runs' / 'test_best'),
                   '--device', config['device']]
    print('Action:', args.action, '| experiment:', config['experiment_id'], flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()

"""Generated-fixture checks of A100 binding, tmux dispatch and report entry points."""

from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import common as c
import compare
import suite


def check_binding(temporary):
    root = temporary/'binding'
    root.mkdir()
    data = root/'data'
    cache = root/'cache'
    cache.mkdir()
    fixed = c.read(ROOT/'membership.json')
    shutil.copyfile(ROOT/'membership.json', root/'membership.json')
    shutil.copyfile(ROOT/'experiment.json', root/'experiment.json')
    import csv
    manifest = root/'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'patient_id', 'split', 'path'])
        writer.writeheader()
        writer.writerows(dict(row, path='old/host/'+row['case_id']+'.npz') for row in fixed)
    for row in fixed:
        if row['split'] == 'test':
            continue
        path = data/'z256/3M'/(row['case_id']+'.npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        groups = ('GT_Layers', 'GT_Capillary', 'GT_LargeVessel') if row['split'] == 'validation' else ('GT_Layers',)
        for group in groups:
            suffix = '.mat' if group == 'GT_Layers' else '.bmp'
            path = root/'annotations'/group/(row['case_id']+suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    c.write(cache/'cache.json', dict(cases={row['case_id']: dict(split=row['split'])
                         for row in fixed if row['split'] != 'test'},
                         protocol=dict(axis_order='ZYX', depth=256, dtype='float32', intensity_scale=255., fov='3M')))
    args = SimpleNamespace(data_root=str(data), manifest=str(manifest), labels_root=None, cache_root=str(cache))
    with patch.object(c, 'ROOT', root), patch.object(c, 'CKPT', root/'checkpoints'):
        with patch.object(np, 'load', side_effect=AssertionError('Binding must not open patient arrays')):
            binding, rows = c.binding(args)
            inputs = c.Inputs(binding, c.read(root/'experiment.json'))
            sample = next(row for row in rows if row['split'] == 'test')
            for action in (lambda: inputs.pair(sample), lambda: inputs.layers(sample['case_id'])):
                try:
                    action()
                except ValueError:
                    pass
                else:
                    raise AssertionError('Test access should fail before file operations')
        assert all(row['path'] == str((data/'z256/3M'/(row['case_id']+'.npz')).resolve()) for row in rows)
        assert len(rows) == 200
        # Header identity changes are rejected by the ordinary re-binding entry.
        (cache/'cache.json').write_text('{}')
        args = SimpleNamespace(data_root=None, manifest=None, labels_root=None, cache_root=None)
        try:
            c.binding(args)
        except ValueError:
            pass
        else:
            raise AssertionError('Changed cache identity accepted')
    return 'Original fixed manifest rebinding, no test file opens, and changed cache rejection'


def check_tmux(temporary):
    roots = [temporary/name for name in c.read(ROOT/'experiment.json')['suite_experiments']]
    for root in roots:
        (root/'verification').mkdir(parents=True)
        c.write(root/'experiment.json', dict(experiment_id=root.name))
        for check in ('preparation.json', 'preflight.json', 'full_preflight.json'):
            c.write(root/'verification'/check, dict(passed=True, source_contract={}, versions=c.versions()))
    init = temporary/'conda.sh'
    init.write_text('# generated fixture\n')
    commands = []

    def launch(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 1 if 'has-session' in command else 0)

    with ExitStack() as stack:
        stack.enter_context(patch.object(suite, 'python', lambda *args: None))
        stack.enter_context(patch.object(suite, 'conda_script', lambda: init))
        stack.enter_context(patch.object(suite, 'tmux_executable', lambda _: '/generated/tmux'))
        stack.enter_context(patch.object(subprocess, 'run', launch))
        stack.enter_context(patch.dict(os.environ, {'CONDA_PREFIX': sys.prefix,
                                                   'CUDA_VISIBLE_DEVICES': 'generated_allocated_gpu'}))
        suite.start(roots, 'lyh_tmux')
        assert sum('new-session' in command for command in commands) == 2
        for root in roots:
            receipt = c.read(root/'verification/latest_launch.json')
            script = Path(receipt['script']).read_text()
            assert 'source ' in script and 'conda activate ' in script and 'run.py train' in script
            assert receipt['python'] == sys.executable and receipt['conda_prefix'] == sys.prefix
            assert 'set -euo pipefail' in script and 'set +e' in script
            assert receipt['gpu_environment']['CUDA_VISIBLE_DEVICES'] == 'generated_allocated_gpu'
            assert 'export CUDA_VISIBLE_DEVICES=generated_allocated_gpu' in script
        for root in roots:
            (root/'checkpoints').mkdir()
            (root/'checkpoints/initial.pt').touch()
        suite.start(roots, 'lyh_tmux')
        assert all(c.read(root/'verification/latest_launch.json')['action'] == 'resume' for root in roots)
        commands.clear()
        with patch.object(subprocess, 'run', lambda command, **kw: subprocess.CompletedProcess(command, 0)):
            suite.start(roots, 'lyh_tmux')
        assert not commands
    return roots, 'Two separate tmux sessions, conda source/activation, resume dispatch and running-session handling'


def check_reports(roots):
    cfg = c.read(ROOT/'experiment.json')
    plane = np.linspace(0, .3, 16*16, dtype=np.float32).reshape(16, 16)
    maps = ('ILM_OPL', 'OPL_BM', 'ILM_OPL_MEAN', 'OPL_BM_MEAN', 'ILM_BM_MEAN', 'FULL', 'ZMAX')
    for index, root in enumerate(roots):
        c.write(root/'experiment.json', dict(suite_experiments=cfg['suite_experiments']))
        c.write(root/'membership.json', c.read(ROOT/'membership.json'))
        digest = ('a' if index == 0 else 'b')*64
        folder = root/'reports/validation/generated'
        (folder/'views').mkdir(parents=True)
        metrics = {key+'_'+metric: (.03-index*.001 if metric == 'mae' else 25+index if metric == 'psnr' else .6+index*.02)
                   for key, _ in compare.OBJECTS for metric in ('mae', 'psnr', 'ssim')}
        cases = [dict(case_id='generated_'+str(i), patient_id='generated_'+str(i), input_sha256=str(i)*64)
                 for i in range(2)]
        c.write(folder/'metrics.json', dict(epoch=5, model_tensor_sha256=digest, split='validation',
                                          patients=2, cases=cases, metrics=metrics))
        c.write(root/'runs/train/status.json', dict(best=dict(epoch=5, model_tensor_sha256=digest,
                                                            report='reports/validation/generated')))
        for case in cases:
            arrays = {**{'gt_'+key: plane for key in maps},
                      **{'pred_'+key: plane*(.8+index*.05) for key in maps}}
            np.savez_compressed(folder/'views'/(case['case_id']+'.npz'), **arrays)
    with patch.object(c, 'ROOT', roots[0]):
        compare.build('validation')
    out = roots[0]/'reports/comparison_validation'
    text = (out/'index.html').read_text(encoding='utf-8')
    assert text.count('<strong>') == 27
    assert (out/'metrics.csv').is_file()
    assert len(list((out/'cases').glob('*.png'))) == 2
    for path in (out/'ssim_comparison.png', *sorted((out/'cases').glob('*.png'))):
        with Image.open(path) as image:
            image.verify()
    shutil.copyfile(out/'ssim_comparison.png', ROOT/'verification/synthetic_comparison.png')
    shutil.copyfile(out/'cases/generated_0.png', ROOT/'verification/synthetic_case_layout.png')
    p = {key: np.tile(plane, (19, 19)) for key in maps}
    g = {key: value.copy() for key, value in p.items()}
    volume = np.broadcast_to(np.linspace(0, .3, 256, dtype=np.float32)[:, None, None], (256, 304, 304))
    c.case_figure(ROOT/'verification/synthetic_multilayer_layout.png', 'GENERATED FIXTURE', p, g, volume, volume)
    return 'Normal comparison entry, 27 bold best cells, paired figures, and all-layer report layout'


def main():
    with tempfile.TemporaryDirectory(prefix='layerproj_runtime_') as temp:
        temporary = Path(temp)
        checks = [check_binding(temporary)]
        roots, message = check_tmux(temporary)
        checks.append(message)
        checks.append(check_reports(roots))
    c.write(ROOT/'verification/runtime_checks.json', dict(passed=True, checks=checks, patient_arrays_used=0,
            real_tmux_sessions_started=0, generated_fixtures_removed=True, source_contract=c.source_contract()))
    print('Runtime entry-point checks passed.', flush=True)


if __name__ == '__main__':
    main()

"""Endpoint supervision, fixed time budgets, best retention and exact CPU resume."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from scipy.io import savemat
from torch import nn
from torch.nn import functional as F

from .model import flow_loss
from .train_start import parse_args, quality, snapshot_best, time_schedule


class RecordingVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(.1))

    def forward(self, state, time, source):
        self.state, self.time, self.source = state.detach(), time.detach(), source.detach()
        return self.weight*source


class StartTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_defaults_and_exact_coupled_time_budget(self):
        args = parse_args(['--manifest', 'data.csv', '--fov', '3M', '--output', 'new',
                           '--axis-order', 'ZYX', '--intensity-scale', '255'])
        self.assertEqual((args.epochs, args.steps_per_epoch, args.save_every, args.start_fraction),
                         (100, 200, 10, .5))
        for epoch in (1, 10, 50, 100):
            uniform, mask0 = time_schedule(2026, epoch, 200, 0.)
            mixed, mask = time_schedule(2026, epoch, 200, .5)
            self.assertFalse(mask0.any())
            self.assertEqual(int(mask.sum()), 100)
            self.assertTrue((uniform > 0).all() and (uniform < 1).all())
            np.testing.assert_array_equal(mixed[~mask], uniform[~mask])
            np.testing.assert_array_equal(mixed[mask], np.zeros(100))
            np.testing.assert_array_equal(mixed, time_schedule(2026, epoch, 200, .5)[0])

    def test_zero_time_loss_and_gradient_equal_direct_reconstruction(self):
        source = torch.rand(1, 1, 8, 8, 8)
        target = torch.rand_like(source)
        model = RecordingVelocity()
        t = torch.zeros(1)
        flow = flow_loss(model, source, target, t)
        torch.testing.assert_close(model.state, source, rtol=0, atol=0)
        torch.testing.assert_close(model.source, source, rtol=0, atol=0)
        flow_grad, = torch.autograd.grad(flow, model.weight)
        direct = F.mse_loss(source+model(source, t, source), target)
        direct_grad, = torch.autograd.grad(direct, model.weight)
        torch.testing.assert_close(flow, direct)
        torch.testing.assert_close(flow_grad, direct_grad)

    def test_best_snapshot_uses_historical_best_not_current_model(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            payload = dict(epoch=20, updates=4000, model={'weight': torch.tensor(20.)},
                best_model={'weight': torch.tensor(13.)}, best=dict(epoch=13, metric='zmax_ssim', value=.8),
                config=dict(save_every=10, epochs=200, val_every=1), manifest=[])
            snapshot_best(root, payload)
            saved = torch.load(root/'best_through_0020.pt', weights_only=False)
            self.assertEqual(saved['epoch'], 13)
            self.assertEqual(float(saved['model']['weight']), 13.)
            metadata = json.loads((root/'best_through_0020.json').read_text())
            self.assertEqual((metadata['through_epoch'], metadata['best_epoch']), (20, 13))
            payload['epoch'], payload['model']['weight'] = 30, torch.tensor(30.)
            snapshot_best(root, payload)
            self.assertEqual(torch.load(root/'best_through_0030.pt', weights_only=False)['epoch'], 13)

    def test_layer_scores_use_same_ground_truth_boundaries(self):
        target = torch.rand(1, 1, 8, 8, 8)
        layers = np.broadcast_to(np.array([1, 100, 300, 400, 500, 640], dtype=np.float32)[:, None, None],
                                 (6, 8, 8)).copy()
        metrics, _ = quality(target.clone(), target, layers)
        self.assertAlmostEqual(metrics['ilm_opl_ssim'], 1.)
        self.assertAlmostEqual(metrics['ilm_opl_pcc'], 1.)
        self.assertEqual(metrics['ilm_opl_empty_rays'], 0)

    def test_cli_resume_counts_best_and_evaluator_compatibility(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            layers_dir = root/'layers'
            layers_dir.mkdir()
            rows = []
            rng = np.random.default_rng(71)
            layers = np.broadcast_to(np.array([1, 100, 300, 400, 500, 640], dtype=np.float32)[:, None, None],
                                     (6, 8, 8)).copy()
            for i in range(200):
                source = rng.random((8, 8, 8), dtype=np.float32)
                np.savez(root/f'{i}.npz', oct=source, octa=source*.3, axis_order=np.array('ZYX'))
                phase = 'train' if i < 140 else 'validation' if i < 150 else 'test'
                rows.append(dict(case_id=str(i), patient_id=str(i), path=f'{i}.npz', split=phase))
                if i == 0 or phase == 'validation':
                    savemat(layers_dir/f'{i}.mat', {'Layer': layers})
            manifest = root/'split.csv'
            with manifest.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            common = ['-m', 'fm3d.train_start', '--manifest', manifest, '--fov', '3M',
                '--axis-order', 'ZYX', '--intensity-scale', '1', '--depth', '8',
                '--patch', '8', '8', '8', '--base-channels', '4', '--steps-per-epoch', '2',
                '--ode-steps', '2', '--val-every', '1', '--save-every', '2',
                '--layers-dir', layers_dir, '--device', 'cpu', '--no-amp']
            environment = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            def run(*args, expect_success=True):
                result = subprocess.run([sys.executable, *map(str, args)], capture_output=True,
                    text=True, env=environment, timeout=150)
                if expect_success:
                    self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result
            run(*common, '--output', root/'resumed', '--epochs', '1')
            # An uncommitted next-epoch trace must be rolled back during resume.
            with (root/'resumed/steps.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(epoch=2, update=999))+'\n')
                stream.write('{"epoch":2,"update":')
            # latest is committed before history; recover a missing final history line too.
            (root/'resumed/history.jsonl').unlink()
            run(*common, '--output', root/'resumed', '--epochs', '2', '--resume')
            run(*common, '--output', root/'continuous', '--epochs', '2')
            resumed = torch.load(root/'resumed/latest.pt', weights_only=False)
            continuous = torch.load(root/'continuous/latest.pt', weights_only=False)
            self.assertEqual(resumed['updates'], 4)
            self.assertEqual(resumed['best']['metric'], 'ilm_opl_ssim')
            for name in resumed['model']:
                torch.testing.assert_close(resumed['model'][name], continuous['model'][name], rtol=0, atol=0)
            records = [json.loads(line) for line in (root/'resumed/steps.jsonl').read_text().splitlines()]
            self.assertEqual([r['update'] for r in records], [1, 2, 3, 4])
            self.assertEqual(sum(r['time']==0 for r in records), 2)
            metadata = json.loads((root/'resumed/best_through_0002.json').read_text())
            self.assertEqual(metadata['best_epoch'], resumed['best']['epoch'])
            rejected = run(*common, '--output', root/'resumed', '--epochs', '3', '--resume',
                           '--start-fraction', '0', expect_success=False)
            self.assertIn('Resume protocol changed', rejected.stderr)
            run('-m', 'fm3d.evaluate', '--checkpoint', root/'resumed/best_through_0002.pt',
                '--split', 'validation', '--output', root/'predictions', '--device', 'cpu')
            self.assertEqual(len(list((root/'predictions').glob('*.npz'))), 10)


if __name__ == '__main__':
    unittest.main()

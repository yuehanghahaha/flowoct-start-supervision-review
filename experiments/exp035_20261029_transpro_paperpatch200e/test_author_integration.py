"""Synthetic cache equivalence, upstream integrity and complete-boundary resume checks."""
import difflib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

import numpy as np
import torch
from torch import nn

from experiment import ROOT, read, sha, write, precision
sys.path.insert(0, str(ROOT / "source"))
from data.base_dataset3d import get_transform
from models import networks
import experiment
import training_bridge as bridge
import run

CONFIG = read(ROOT / "experiment.json")


class TinyAuthorModel:
    def __init__(self):
        self.save_dir = str(bridge.ROOT / "checkpoints/author_log/TINY")
        for name in bridge.MODEL_NAMES:
            setattr(self, name, nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2)))
        self.optimizers = [torch.optim.Adam(self.netG.parameters(), lr=.0002, betas=(.5, .999)),
                           torch.optim.Adam(list(self.netD.parameters()) + list(self.netD_proj.parameters()), lr=.0002, betas=(.5, .999))]
        options = SimpleNamespace(lr_policy="linear", epoch_count=1, n_epochs=100, n_epochs_decay=100)
        self.schedulers = [networks.get_scheduler(item, options) for item in self.optimizers]

    def save(self, label):
        directory = Path(self.save_dir)
        directory.mkdir(parents=True, exist_ok=True)
        for name, suffix in bridge.AuthorSession.labels.items():
            torch.save(getattr(self, name).state_dict(), directory / f"{label}_net_{suffix}.pth")


class IntegrationTests(unittest.TestCase):
    def test_only_declared_paper_model_patch_differs(self):
        manifest = read(ROOT / "provenance/upstream_sha256.json")
        self.assertEqual(len(manifest), 65)
        patch_record = read(ROOT / "provenance/paper_source_patch.json")
        for name, expected in manifest.items():
            if name == patch_record["file"]:
                self.assertEqual(sha(ROOT / "source" / name), patch_record["patched_sha256"])
                self.assertEqual(sha(ROOT / "provenance/TransPro_model.upstream.py"), expected)
                continue
            self.assertEqual(sha(ROOT / "source" / name), expected, name)

    def test_author_loop_only_has_declared_integration_edits(self):
        original = (ROOT / "source/train3d.py").read_text(encoding="utf-8")
        integrated = (ROOT / "train_author.py").read_text(encoding="utf-8")
        delta = "".join(difflib.unified_diff(original.splitlines(True), integrated.splitlines(True),
                       fromfile="source/train3d.py", tofile="train_author.py"))
        self.assertEqual(delta, (ROOT / "provenance/train_integration.diff").read_text(encoding="utf-8"))
        self.assertIn("model.optimize_parameters()", integrated)
        self.assertIn("mae = MAE_(fake_B,real_B)", integrated)
        self.assertNotIn("model.eval()", integrated)
        self.assertNotIn("eight_metrics", integrated)
        self.assertNotIn("finite_gradients", integrated)
        self.assertIn("if epoch > opt.n_epochs:\n        model.update_learning_rate()", integrated)

    def test_cached_crop_matches_released_torchio_transform(self):
        # Full real geometry, synthetic arrays only; independent of any patient.
        rng = np.random.default_rng(51)
        array = rng.random((1, 256, 304, 304), dtype=np.float32)
        params = {"crop_pos": (7, 41, 0, 0, 19, 29), "flip": True}
        options = SimpleNamespace(preprocess="resize_and_crop", load_size=304, crop_size=256, no_flip=True)
        raw = np.ascontiguousarray(array[0].transpose(1, 0, 2)[None])
        expected = get_transform(options, params)(raw)
        expected = 2 * (expected - expected.min()) / (expected.max() - expected.min()) - 1
        actual = bridge.cache_crop(array, params)
        self.assertEqual(tuple(actual.shape), (1, 256, 256, 256))
        self.assertLessEqual(float(np.max(np.abs(expected - actual.numpy()))), 2e-7)

    def test_precision_does_not_override_framework_flags(self):
        flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                 torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
        precision(CONFIG)
        self.assertEqual(flags, (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic))
        self.assertEqual(torch.get_default_dtype(), torch.float32)

    def test_best_and_resume_survive_later_best_overwrites(self):
        with TemporaryDirectory(dir=ROOT / "verification", prefix="synthetic_resume_") as directory:
            path = Path(directory)
            with patch.object(experiment, "ROOT", path), patch.object(bridge, "ROOT", path), patch.object(run, "protocol_id", return_value="SYNTHETIC"):
                session, model = bridge.AuthorSession(), TinyAuthorModel()
                session.attach(model, SimpleNamespace(save_epoch_freq=5))
                expected = None
                for epoch in range(1, 8):
                    session.begin_epoch(epoch)
                    with torch.no_grad():
                        model.netG[0].weight.fill_(float(epoch))
                        model.netG_t[1].running_mean.fill_(epoch / 10)
                    for optimizer in model.optimizers:
                        optimizer.zero_grad()
                        for group in optimizer.param_groups:
                            for parameter in group["params"]:
                                parameter.grad = torch.ones_like(parameter)
                        optimizer.step()
                    session.training_finished()
                    session.begin_validation()
                    session.observe_validation(epoch, 1 / epoch)
                    session.save(model, "best")
                    session.epoch_complete(epoch, model, 1 / epoch, epoch * 140)
                    if epoch == 5:
                        expected = model.netG[0].weight.detach().clone()
                restored, replacement = bridge.AuthorSession(), TinyAuthorModel()
                restored.attach(replacement, SimpleNamespace(save_epoch_freq=5))
                self.assertEqual(restored.start_epoch, 6)
                self.assertEqual(restored.total_steps, 700)
                self.assertEqual(restored.best["epoch"], 5)
                self.assertTrue(torch.equal(replacement.netG[0].weight, expected))
                self.assertTrue(torch.allclose(replacement.netG_t[1].running_mean, torch.full((2,), .5)))
                self.assertEqual(int(next(iter(replacement.optimizers[0].state.values()))["step"]), 5)
                self.assertEqual(read(path / "checkpoints/best.json")["epoch"], 5)

    def test_author_options_and_worker_count(self):
        with patch.object(run, "read", return_value={"cache": "/SYNTHETIC_CACHE"}):
            argv = run.author_argv("-1")
        self.assertEqual(argv[argv.index("--lambda_C") + 1], "5")
        self.assertEqual(argv[argv.index("--num_threads") + 1], "4")
        self.assertNotIn("--no_lsgan", argv)
        self.assertEqual(CONFIG["vpg_representation"], "paper_raw_logits")
        self.assertEqual(CONFIG["gan_objective"], "paper_logarithmic_minimax_keep_author_D_half")
        from options.train_options import TrainOptions
        with patch.object(sys, "argv", argv), patch.object(TrainOptions, "print_options"):
            options = TrainOptions().parse()
        self.assertEqual((options.ngf, options.ndf, options.crop_size, options.batch_size), (64, 64, 256, 1))
        self.assertEqual((options.lambda_A, options.lambda_C, options.beta1, options.lr), (10., 5., .5, .0002))
        self.assertEqual((options.n_epochs, options.n_epochs_decay, options.val_epoch_freq, options.save_epoch_freq), (100, 100, 1, 5))
        self.assertFalse(options.no_lsgan)
        self.assertTrue(options.no_flip)
        with patch.object(bridge, "CachedAuthorDataset", return_value=torch.utils.data.TensorDataset(torch.zeros((4, 2)))):
            loader = bridge.create_dataset(SimpleNamespace(batch_size=1, serial_batches=False, num_threads=4), "val")
        self.assertEqual(loader.num_workers, 4)
        self.assertIsInstance(loader.sampler, torch.utils.data.RandomSampler)

    def test_test_patients_not_exposed_to_training(self):
        with patch.object(bridge, "Inputs", side_effect=AssertionError("must not read test")):
            with self.assertRaises(ValueError):
                bridge.CachedAuthorDataset(None, "test")

    def test_released_scheduler_timing(self):
        parameter = nn.Parameter(torch.tensor(0.))
        optimizer = torch.optim.Adam([parameter], lr=.0002)
        options = SimpleNamespace(lr_policy="linear", epoch_count=1, n_epochs=100, n_epochs_decay=100)
        scheduler = networks.get_scheduler(optimizer, options)
        for epoch in range(1, 201):
            self.assertEqual(optimizer.param_groups[0]["lr"], .0002)
            optimizer.step()
            if epoch > 100:
                scheduler.step()


if __name__ == "__main__":
    (ROOT / "verification").mkdir(exist_ok=True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IntegrationTests))
    write(ROOT / "verification/local_author_tests.json", {"tests": result.testsRun, "success": result.wasSuccessful(),
          "synthetic_only": True, "full_width_A100_backward_verified": False, "patient_arrays_opened": False})
    raise SystemExit(0 if result.wasSuccessful() else 1)

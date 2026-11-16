"""Synthetic numerical/gradient/coverage checks; no clinical test patients or GPU training."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import copy
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from experiment import ROOT, read, write, precision
from dataset import Inputs, crop, unit
from metrics import image_metrics, eight_metrics, density_metrics, aggregate
from network import TransPro, generator_minimax, generator_losses, networks_2g_st, networks3d, networks
from evaluation import predict_full, projection_figure, build_report
from run import learning_rate

CONFIG = read(ROOT / "experiment.json")


class PipelineTests(unittest.TestCase):
    def test_heldout_arrays_and_labels_are_guarded(self):
        inputs = Inputs.__new__(Inputs)
        row = {"case_id": "SYNTHETIC_HELDOUT", "split": "test"}
        with patch("numpy.load", side_effect=AssertionError("must not read heldout data")):
            with self.assertRaises(ValueError):
                inputs.pair(row)
        with patch("PIL.Image.open", side_effect=AssertionError("must not read heldout labels")):
            with self.assertRaises(ValueError):
                inputs.mask(row)

    def test_original_architecture_meta_shape(self):
        with torch.device("meta"):
            generator = networks_2g_st.generator3D(64)
            result = generator(torch.empty(1, 1, 256, 256, 256))
            self.assertEqual(tuple(result.shape), (1, 1, 256, 256, 256))
            self.assertEqual(sum(p.numel() for p in generator.parameters()), 217607169)
            d3 = networks3d.NLayerDiscriminator(2, 64, norm_layer=nn.BatchNorm3d, gpu_ids=[])
            d2 = networks.NLayerDiscriminator(2, 64, norm_layer=nn.BatchNorm2d)
            self.assertEqual(tuple(d3(torch.empty(1, 2, 256, 256, 256)).shape), (1, 1, 30, 30, 30))
            self.assertEqual(tuple(d2(torch.empty(1, 2, 256, 256)).shape), (1, 1, 30, 30))

    def test_log_minimax_gradient_and_frozen_teacher_input_gradient(self):
        score = torch.tensor([-20., 0., 20.], requires_grad=True)
        generator_minimax(score).backward()
        self.assertTrue(torch.allclose(score.grad, -torch.sigmoid(score.detach()) / 3))
        teacher = nn.Conv2d(1, 2, 1).requires_grad_(False)
        fake = torch.randn(1, 1, 16, 16, requires_grad=True)
        torch.nn.functional.l1_loss(teacher(fake), torch.zeros(1, 2, 16, 16)).backward()
        self.assertGreater(float(fake.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))

    def test_separate_paper_weights_and_gradient(self):
        fake = torch.ones(1, 1, 16, 16, requires_grad=True)
        zero = torch.zeros_like(fake)
        terms, total = generator_losses(fake, zero, fake, zero, zero, fake, zero, zero, zero, CONFIG["loss_weights"])
        self.assertEqual([float(terms[k].detach()) for k in ("volume", "projection", "vpg", "hcg")], [10., 10., 5., 5.])
        total.backward()
        self.assertAlmostEqual(float(fake.grad.sum()), 30., places=5)

    def test_weighted_images_square_background_scaling(self):
        target = np.zeros((32, 32), np.float32)
        pred = np.ones_like(target)
        original = image_metrics(pred, target, CONFIG)
        weighted = image_metrics(.1 * pred, .1 * target, CONFIG)
        self.assertAlmostEqual(weighted["MAE"], .1, places=7)
        self.assertAlmostEqual(weighted["PSNR"] - original["PSNR"], 20., places=5)

    def test_complete_update_and_optimizer_resume_on_synthetic_fixture(self):
        class Fixture(TransPro):
            def __init__(self):
                nn.Module.__init__(self)
                self.G = nn.Sequential(nn.Conv3d(1, 1, 1), nn.Tanh())
                self.D3 = nn.Conv3d(2, 1, 1)
                self.D2 = nn.Conv2d(2, 1, 1)
                self.H = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Tanh()).requires_grad_(False)
                self.S = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2)).requires_grad_(False)
                self.train(True)
        model = Fixture()
        optimizer_g, optimizer_d = model.optimizers(CONFIG)
        a, b = torch.randn(1, 1, 16, 16, 16), torch.randn(1, 1, 16, 16, 16)
        frozen_stats = model.S[1].running_mean.clone()
        model.step(a, b, optimizer_g, optimizer_d, CONFIG["loss_weights"])
        clone = copy.deepcopy(model)
        with TemporaryDirectory(dir=ROOT / "verification", prefix="synthetic_resume_") as directory:
            path = Path(directory) / "fixture.pt"
            torch.save({"models": model.trained_state(), "g": optimizer_g.state_dict(), "d": optimizer_d.state_dict()}, path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for name, state in payload["models"].items():
                getattr(clone, name).load_state_dict(state)
            g2, d2 = clone.optimizers(CONFIG)
            g2.load_state_dict(payload["g"])
            d2.load_state_dict(payload["d"])
            left = model.step(a, b, optimizer_g, optimizer_d, CONFIG["loss_weights"])
            right = clone.step(a, b, g2, d2, CONFIG["loss_weights"])
            self.assertEqual(left, right)
            for p, q in zip(model.G.parameters(), clone.G.parameters()):
                self.assertTrue(torch.equal(p, q))
        self.assertTrue(torch.equal(model.S[1].running_mean, frozen_stats))
        self.assertTrue(all(p.grad is None for teacher in (model.H, model.S) for p in teacher.parameters()))

    def test_bscan_psnr_and_eight_metric_contract(self):
        rng = np.random.default_rng(5)
        target = rng.uniform(.1, .8, (32, 32, 32)).astype(np.float32)
        pred = target.copy()
        pred[:, :16] += .01
        pred[:, 16:] += .1
        mask = rng.random((32, 32)) > .5
        values, slices, views = eight_metrics(pred, target, mask, CONFIG)
        self.assertEqual(len(values), 8)
        expected = np.mean([row["PSNR"] for row in slices])
        whole_volume = 10 * np.log10(1 / np.mean((pred - target) ** 2))
        self.assertAlmostEqual(values["PSNR"], expected)
        self.assertGreater(abs(values["PSNR"] - whole_volume), 1)
        self.assertEqual(views["target_yx"].shape, (32, 32))

    def test_vdc_undefined_and_patient_macro(self):
        values, _ = density_metrics(np.ones((32, 32)), np.ones((32, 32)))
        self.assertIsNone(values["VDC"])
        rows = [{name: 1. for name in CONFIG["reference"]["paper_table3M"]}, {name: 3. for name in CONFIG["reference"]["paper_table3M"]}]
        self.assertEqual(aggregate(rows)["patient_mean"]["MAE"], 2.)

    def test_full_view_coverage_and_float_prediction(self):
        class Echo(nn.Module):
            def forward(self, x):
                return x
        y = np.linspace(0, 1, 304, dtype=np.float32)[None, None, :, None]
        z = np.linspace(0, 1, 256, dtype=np.float32)[None, :, None, None]
        x = np.linspace(0, 1, 304, dtype=np.float32)[None, None, None, :]
        source = np.ascontiguousarray((y + z + x) / 3)
        pred = predict_full(Echo(), source, torch.device("cpu"))
        self.assertEqual(pred.shape, (256, 304, 304))
        self.assertTrue(np.isfinite(pred).all())
        self.assertEqual(pred.dtype, np.float32)
        self.assertGreater(float(pred[:, 303, 303].mean()), float(pred[:, 0, 0].mean()))
        adapted = crop(source, 7, 19)
        self.assertEqual(adapted.shape, (1, 1, 256, 256, 256))

    def test_explicit_lr_schedule_and_synthetic_figure(self):
        self.assertEqual(learning_rate(100, CONFIG), .0002)
        self.assertEqual(learning_rate(101, CONFIG), .0002)
        self.assertAlmostEqual(learning_rate(200, CONFIG), .000002)
        rng = np.random.default_rng(11)
        target = rng.uniform(0, .9, (32, 32, 32)).astype(np.float32)
        spatial_error = .025 * np.sin(np.linspace(0, 2 * np.pi, 32, dtype=np.float32))[None, None, :]
        pred = np.clip(.85 * target + .04 + spatial_error, 0, 1)
        values, _, views = eight_metrics(pred, target, rng.random((32, 32)) > .6, CONFIG)
        preview = ROOT / "verification/synthetic_preview.png"
        projection_figure(unit(target.mean(0)), views, "SYNTHETIC", preview, synthetic=True)
        with TemporaryDirectory(dir=ROOT / "verification", prefix="synthetic_report_") as directory:
            path = Path(directory)
            (path / "figures").mkdir()
            row = {"case_id": "SYNTHETIC", "patient_id": "SYNTHETIC", **values}
            build_report(path, [row], aggregate([row]), CONFIG, "synthetic", 0)
            self.assertTrue((path / "paper_comparison.csv").is_file())
            self.assertTrue((path / "index.html").is_file())


if __name__ == "__main__":
    precision(CONFIG, CONFIG["seed"])
    (ROOT / "verification").mkdir(exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PipelineTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write(ROOT / "verification/local_pipeline.json", {"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
          "success": result.wasSuccessful(), "synthetic_only": True, "no_clinical_test_data": True,
          "actual_full_width_GPU_backward_verified": False, "scope": "Numerical metrics, gradients, original architecture meta shapes and complete-view coverage; not A100 acceptance"})
    raise SystemExit(0 if result.wasSuccessful() else 1)

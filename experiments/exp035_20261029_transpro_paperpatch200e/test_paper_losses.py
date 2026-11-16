"""Check paper loss equations and protect unrelated author methods; synthetic data only."""
import ast
from pathlib import Path
import sys
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from experiment import ROOT, write
sys.path.insert(0, str(ROOT / "source"))
from models.TransPro_model import TransProModel
from models.networks_2g_st import GANLoss
from util.image_pool import ImagePool


def log_criterion():
    criterion = GANLoss(use_lsgan=False)
    criterion.loss = nn.BCEWithLogitsLoss()
    return criterion


class PaperLossTests(unittest.TestCase):
    def test_log_minimax_values_and_gradients(self):
        scores = torch.tensor([-80., -2., 0., 2., 80.], dtype=torch.float64, requires_grad=True)
        criterion = log_criterion().double()
        generated = -criterion(scores, False)
        self.assertTrue(torch.isfinite(generated))
        self.assertTrue(torch.allclose(generated, F.logsigmoid(-scores).mean(), atol=1e-14))
        generated.backward()
        self.assertTrue(torch.allclose(scores.grad, -scores.detach().sigmoid()/scores.numel(), atol=1e-14))
        self.assertTrue(torch.allclose(criterion(scores, True), F.softplus(-scores).mean(), atol=1e-14))
        self.assertTrue(torch.allclose(criterion(scores, False), F.softplus(scores).mean(), atol=1e-14))

    def test_patched_author_update_uses_projection10_and_raw_logits(self):
        torch.manual_seed(51)  # Synthetic test only; production seeding remains author behavior.
        model = TransProModel.__new__(TransProModel)
        from types import SimpleNamespace
        model.opt = SimpleNamespace(lambda_A=10., lambda_C=5., direction="AtoB")
        model.device = torch.device("cpu")
        model.netG = nn.Sequential(nn.Conv3d(1, 1, 1), nn.Tanh())
        model.netD = nn.Conv3d(2, 1, 1)
        model.netD_proj = nn.Conv2d(2, 1, 1)
        model.netG_t = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Tanh())
        model.net = nn.Conv2d(1, 2, 1)
        for teacher in (model.netG_t, model.net):
            teacher.requires_grad_(False)
        model.fake_AB_pool = ImagePool(0)
        model.criterionGAN = log_criterion()
        model.criterionL1 = nn.L1Loss()
        model.optimizer_G = torch.optim.Adam(model.netG.parameters(), lr=.0002, betas=(.5, .999))
        model.optimizer_D = torch.optim.Adam(list(model.netD.parameters()) + list(model.netD_proj.parameters()), lr=.0002, betas=(.5, .999))
        model.set_input({"A": torch.rand((1,1,4,4,4))*2-1, "B": torch.rand((1,1,4,4,4))*2-1, "A_paths":["SYNTHETIC"]})
        before = model.netG[0].weight.detach().clone()
        model.optimize_parameters()  # Executes the actual patched author methods.
        with torch.no_grad():
            score3 = model.netD(torch.cat((model.real_A, model.fake_B), 1))
            score2 = model.netD_proj(torch.cat((model.real_A_proj, model.fake_B_proj_s), 1))
            raw_fake = model.net((model.fake_B_proj_s + 1)/2)[0]
            raw_real = model.net((model.real_B_proj + 1)/2)[0]
            self.assertTrue(torch.allclose(model.loss_G_GAN, F.logsigmoid(-score3).mean()))
            self.assertTrue(torch.allclose(model.loss_G_GAN_proj, F.logsigmoid(-score2).mean()))
            self.assertTrue(torch.allclose(model.loss_G_L1_pm, 10*F.l1_loss(model.fake_B_proj_s, model.real_B_proj)))
            self.assertTrue(torch.allclose(model.loss_G_L1_seg, 5*F.l1_loss(raw_fake, raw_real)))
            self.assertTrue(torch.allclose(model.loss_G_L1_pm_st, 5*F.l1_loss(model.fake_B_proj_s, model.fake_B_proj_t)))
        self.assertFalse(torch.equal(before, model.netG[0].weight))
        self.assertTrue(model.netG_t.training and model.net.training)
        self.assertTrue(all(p.grad is None for teacher in (model.netG_t, model.net) for p in teacher.parameters()))

    def test_unrelated_author_methods_are_structurally_identical(self):
        def methods(path):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TransProModel")
            return {node.name:ast.dump(node, include_attributes=False) for node in cls.body if isinstance(node, ast.FunctionDef)}
        original = methods(ROOT / "provenance/TransPro_model.upstream.py")
        patched = methods(ROOT / "source/models/TransPro_model.py")
        self.assertEqual(set(original), set(patched))
        changed = {name for name in original if original[name] != patched[name]}
        self.assertEqual(changed, {"__init__", "backward_G"})
        self.assertEqual(original["backward_D"], patched["backward_D"])
        self.assertEqual(original["optimize_parameters"], patched["optimize_parameters"])


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PaperLossTests))
    write(ROOT / "verification/local_paper_loss_tests.json", {"tests":result.testsRun,"success":result.wasSuccessful(),
          "synthetic_only":True,"actual_author_loss_methods_checked":True,"original_width_GPU_update_checked":False})
    raise SystemExit(0 if result.wasSuccessful() else 1)

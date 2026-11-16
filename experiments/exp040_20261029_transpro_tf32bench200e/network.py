"""Official network architectures with explicitly separated paper loss coefficients."""
import sys

import torch
from torch import nn
import torch.nn.functional as F

from experiment import ROOT

sys.path.insert(0, str(ROOT / "source"))
from models import networks_2g_st, networks3d, networks
from unet import UNet


def projection(volume):
    value = volume.mean(dim=3)
    low = value.amin(dim=(2, 3), keepdim=True)
    high = value.amax(dim=(2, 3), keepdim=True)
    if bool(torch.any(high <= low)):
        raise FloatingPointError("Constant projection; official min-max is undefined")
    return 2 * (value - low) / (high - low) - 1


def generator_minimax(logits):
    # Eq. (13) minimizes log(1-D(G(X))); this is deliberately not LSGAN/non-saturating BCE.
    return -F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))


def generator_losses(fake, real, fake_projection, real_projection, hcg_target, fake_logits, real_logits, score3, score2, weights):
    terms = {"adv3": generator_minimax(score3), "adv2": generator_minimax(score2),
             "volume": F.l1_loss(fake, real) * weights["volume"],
             "projection": F.l1_loss(fake_projection, real_projection) * weights["projection"],
             "hcg": F.l1_loss(fake_projection, hcg_target) * weights["hcg"],
             "vpg": F.l1_loss(fake_logits, real_logits) * weights["vpg"]}
    return terms, sum(terms.values())


def finite_gradients(parameters):
    for parameter in parameters:
        if parameter.grad is not None and (parameter.grad.dtype != torch.float32 or not bool(torch.isfinite(parameter.grad).all())):
            raise FloatingPointError("A gradient is not finite FP32")


class TransPro(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        if config["ngf"] != 64 or config["ndf"] != 64:
            raise ValueError("The formal architecture must retain official width64")
        self.G = networks_2g_st.generator3D(64)
        self.G.weight_init(mean=0.0, std=0.02)
        self.D3 = networks3d.NLayerDiscriminator(2, 64, n_layers=3, norm_layer=nn.BatchNorm3d, use_sigmoid=False, gpu_ids=[])
        self.D3.apply(networks3d.weights_init)
        self.D2 = networks.NLayerDiscriminator(2, 64, n_layers=3, norm_layer=nn.BatchNorm2d)
        networks.init_weights(self.D2, init_type="normal", init_gain=0.02)
        self.H = networks_2g_st.generator(64)
        self.S = UNet(n_channels=1, n_classes=2, bilinear=False)
        self.H.load_state_dict(torch.load(ROOT / "pretrain_weights/hcg.pth", map_location="cpu", weights_only=True))
        self.S.load_state_dict(torch.load(ROOT / "pretrain_weights/vpg.pth", map_location="cpu", weights_only=True))
        for teacher in (self.H, self.S):
            teacher.requires_grad_(False)
            teacher.eval()
        self.to(device=device, dtype=torch.float32)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        self.H.eval()
        self.S.eval()
        return self

    def optimizers(self, config):
        options = dict(lr=config["learning_rate"], betas=tuple(config["betas"]), weight_decay=0)
        return torch.optim.Adam(self.G.parameters(), **options), torch.optim.Adam(list(self.D3.parameters()) + list(self.D2.parameters()), **options)

    def step(self, source, target, optimizer_g, optimizer_d, weights):
        if source.dtype != torch.float32 or target.dtype != torch.float32:
            raise ValueError("Expected FP32 inputs")
        self.train(True)
        optimizer_g.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        fake = self.G(source)
        ap, bp, fp = [projection(value) for value in (source, target, fake)]
        self.D3.requires_grad_(True)
        self.D2.requires_grad_(True)
        d_loss = 0
        for discriminator, a, b, f in ((self.D3, source, target, fake), (self.D2, ap, bp, fp)):
            real_score = discriminator(torch.cat((a, b), dim=1))
            fake_score = discriminator(torch.cat((a, f.detach()), dim=1))
            d_loss = d_loss + F.binary_cross_entropy_with_logits(real_score, torch.ones_like(real_score))
            d_loss = d_loss + F.binary_cross_entropy_with_logits(fake_score, torch.zeros_like(fake_score))
        if not bool(torch.isfinite(d_loss)):
            raise FloatingPointError("Discriminator loss is not finite")
        d_loss.backward()
        finite_gradients([*self.D3.parameters(), *self.D2.parameters()])
        optimizer_d.step()
        optimizer_d.zero_grad(set_to_none=True)
        self.D3.requires_grad_(False)
        self.D2.requires_grad_(False)
        with torch.no_grad():
            teacher_projection = self.H(ap)
            real_logits = self.S((bp + 1) / 2)
        fake_logits = self.S((fp + 1) / 2)  # Frozen teacher still passes the gradient to the generated image.
        score3 = self.D3(torch.cat((source, fake), dim=1))
        score2 = self.D2(torch.cat((ap, fp), dim=1))
        terms, g_loss = generator_losses(fake, target, fp, bp, teacher_projection, fake_logits, real_logits, score3, score2, weights)
        if not bool(torch.isfinite(g_loss)):
            raise FloatingPointError("Generator loss is not finite")
        g_loss.backward()
        finite_gradients(self.G.parameters())
        optimizer_g.step()
        optimizer_g.zero_grad(set_to_none=True)
        if any(p.grad is not None for teacher in (self.H, self.S) for p in teacher.parameters()):
            raise RuntimeError("A frozen teacher accumulated gradients")
        return {"D": float(d_loss.detach()), "G": float(g_loss.detach()), **{k: float(v.detach()) for k, v in terms.items()}}

    def trained_state(self):
        return {name: {key: value.detach().cpu() for key, value in getattr(self, name).state_dict().items()} for name in ("G", "D3", "D2")}


def inference_generator(state, device):
    generator = networks_2g_st.generator3D(64)
    generator.load_state_dict(state)
    return generator.to(device=device, dtype=torch.float32).eval()

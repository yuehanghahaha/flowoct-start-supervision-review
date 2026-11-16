"""OCT-start paired flow matching with a single image channel."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from fm3d.inference import generate_volume
from fm3d.model import VelocityUNet3D


def make_model(cfg):
    # Keep the exp017 random draws, including the state half of the input kernel.
    model = VelocityUNet3D(cfg['base_channels'], condition=True)
    parent_stem = model.stem
    stem = nn.Conv3d(1, parent_stem.out_channels, kernel_size=3, padding=1)
    with torch.no_grad():
        stem.weight.copy_(parent_stem.weight[:, :1])
        stem.bias.copy_(parent_stem.bias)
    model.stem = stem
    model.condition = False
    return model


def make_optimizer(model, cfg):
    return torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])


def velocity_and_loss(model, source, target, t, amp):
    fraction = t[:, None, None, None, None]
    state = (1-fraction) * source + fraction * target
    with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16,
                        enabled=amp and state.device.type == 'cuda'):
        velocity = model(state, t)
    loss = F.mse_loss(velocity.float(), target - source)
    return velocity, loss


def update(model, optimizer, source, target, time, cfg):
    device = next(model.parameters()).device
    source, target = source.to(device), target.to(device)
    t = torch.tensor([time], dtype=torch.float32, device=device)
    optimizer.zero_grad(set_to_none=True)
    velocity, loss = velocity_and_loss(model, source, target, t, cfg['amp'])
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
    loss_value, grad_norm = torch.stack((loss.detach(), norm)).tolist()
    if not math.isfinite(loss_value) or not math.isfinite(grad_norm) or grad_norm <= 0:
        raise FloatingPointError('Invalid flow matching loss or gradient')
    optimizer.step()
    return dict(fm_mse=loss_value, gradient_norm=grad_norm, velocity_dtype=str(velocity.dtype))


def predict(model, source, cfg):
    return generate_volume(model, source, cfg['patch'], cfg['ode_steps'], cfg['overlap'], cfg['amp'])

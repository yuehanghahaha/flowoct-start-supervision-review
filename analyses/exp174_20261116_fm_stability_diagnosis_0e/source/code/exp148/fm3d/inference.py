"""Euler integration with one full float32 state and overlapping velocities."""
from itertools import product

import torch
from torch.nn import functional as F


def window_starts(length, patch, overlap):
    stride = max(1, round(patch * (1 - overlap)))
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


@torch.inference_mode()
def generate_volume(model, source, patch, steps=8, overlap=0.5, amp=False):
    if source.ndim != 5 or source.shape[:2] != (1, 1):
        raise ValueError('Expected one [1,1,Z,Y,X] OCT volume')
    if steps < 1 or len(patch) != 3 or min(patch) < 8 or not 0 <= overlap < 1:
        raise ValueError('Invalid Euler steps or sliding-window geometry')

    device = next(model.parameters()).device
    shape = source.shape[2:]
    padding = [v for n, p in reversed(list(zip(shape, patch))) for v in (0, max(0, p-n))]
    # OCT initializes the global state. Every velocity call sees the current crop and t.
    state = F.pad(source.to(device, dtype=torch.float32), padding, mode='replicate')
    starts = [window_starts(n, p, overlap) for n, p in zip(state.shape[2:], patch)]
    regions = [(slice(None), slice(None)) + tuple(slice(s, s+p) for s, p in zip(pos, patch))
               for pos in product(*starts)]
    axes = [torch.hann_window(p, periodic=False).clamp_min(0.05) for p in patch]
    weight = (axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :])[None, None].to(device)
    counts = torch.zeros_like(state)
    for region in regions:
        counts[region] += weight
    velocity = torch.zeros_like(state)
    was_training = model.training
    model.eval()
    try:
        for step in range(steps):
            velocity.zero_()
            time = torch.full((1,), step / steps, device=device, dtype=torch.float32)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == 'cuda',
                                dtype=torch.bfloat16):
                for region in regions:
                    patch_velocity = model(state[region].contiguous(), time)
                    velocity[region] += patch_velocity.float() * weight
            state.add_(velocity.div_(counts), alpha=1 / steps)
        if not torch.isfinite(state).all():
            raise FloatingPointError('Nonfinite full-volume Euler state')
        return state[:, :, :shape[0], :shape[1], :shape[2]].clamp_(0, 1).cpu()
    finally:
        model.train(was_training)

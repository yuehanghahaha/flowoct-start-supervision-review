"""Paired straight flow and random-time, raw-endpoint voxel L1."""
import time
import numpy as np
import torch
from torch.nn import functional as F
from fm3d.model import make_model as parent_model
from fm3d.inference import generate_volume


def make_model(cfg):
    return parent_model(cfg['base_channels'], condition=True, representation='voxel')


def optimizers(model, cfg):
    return [torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])]


def batch(inputs, row, cfg, time):
    source, target, origin, slices = inputs.training_patch(row)
    layers = None
    if cfg['projection_mode'] != 'none' and float(time) == 0.:
        layers = inputs.layers(row['case_id'])[:, slices[1], slices[2]]
    return dict(source=source, target=target, layers=layers, time=float(time), origin_ZYX=origin)


@torch.no_grad()
def _boxes(images, ratio):
    """Batched public crop-helper geometry, inclusive coordinates [x0,z0,x1,z1]."""
    n, height, width = images.shape
    threshold = images.amax((1, 2)) * ratio
    z = torch.arange(height, device=images.device)[None]
    x = torch.arange(width, device=images.device)[None]
    rows = images.mean(2) > threshold[:, None]
    has_rows = rows.any(1)
    z0 = torch.where(rows, z, height).amin(1)
    z1 = torch.where(rows, z, -1).amax(1)
    z0 = torch.where(has_rows, z0, 0)
    z1 = torch.where(has_rows, z1, height-1)
    selected_rows = (z >= z0[:, None]) & (z <= z1[:, None])
    columns = (images * selected_rows[:, :, None]).sum(1) / (z1-z0+1)[:, None]
    valid_cols = columns > threshold[:, None]
    has_cols = valid_cols.any(1)
    x0 = torch.where(valid_cols, x, width).amin(1)
    x1 = torch.where(valid_cols, x, -1).amax(1)
    x0 = torch.where(has_cols & has_rows, x0, 0)
    x1 = torch.where(has_cols & has_rows, x1, width-1)
    return torch.stack((x0, z0, x1, z1), 1), ~has_rows


@torch.no_grad()
def content_boxes(endpoint, target, ratio=.05, padding=10):
    if endpoint.shape != target.shape or endpoint.ndim != 5 or endpoint.shape[1] != 1:
        raise ValueError('Expected equal [B,1,Z,Y,X] tensors')
    height, width = endpoint.shape[2], endpoint.shape[4]
    p = endpoint.detach().clamp(0, 1)[:, 0].permute(0, 2, 1, 3).reshape(-1, height, width)
    g = target.detach().clamp(0, 1)[:, 0].permute(0, 2, 1, 3).reshape(-1, height, width)
    pb, pf = _boxes(p, ratio)
    gb, gf = _boxes(g, ratio)
    box = torch.cat((torch.minimum(pb[:, :2], gb[:, :2]),
                     torch.maximum(pb[:, 2:], gb[:, 2:])), 1)
    box[:, 1] = (box[:, 1]-padding).clamp_min(0)
    box[:, 3] = (box[:, 3]+padding).clamp_max(height-1)
    return box, (pf | gf)


def endpoint_l1(endpoint, target, cfg):
    mode = cfg['endpoint_l1_mode']
    if mode == 'none':
        return endpoint.new_zeros(()), dict(roi_fraction=1., roi_fallback_fraction=0.)
    if mode == 'full':
        return F.l1_loss(endpoint, target), dict(roi_fraction=1., roi_fallback_fraction=0.)
    if mode != 'content_box':
        raise ValueError(mode)
    box, fallback = content_boxes(endpoint, target, cfg['content_box_threshold_ratio'], cfg['content_box_z_padding'])
    height, width = endpoint.shape[2], endpoint.shape[4]
    z = torch.arange(height, device=endpoint.device)[None, :, None]
    x = torch.arange(width, device=endpoint.device)[None, None, :]
    mask = ((x >= box[:, 0, None, None]) & (x <= box[:, 2, None, None]) &
            (z >= box[:, 1, None, None]) & (z <= box[:, 3, None, None]))
    area = mask.sum((1, 2))
    error = (endpoint-target).abs()[:, 0].permute(0, 2, 1, 3).reshape(-1, height, width)
    loss = ((error*mask).sum((1, 2)) / area).mean()
    return loss, dict(roi_fraction=float(area.float().mean() / (height*width)),
                      roi_fallback_fraction=float(fallback.float().mean()))


def update(model, opts, data, cfg):
    from common import projection_loss
    tick = time.perf_counter()
    device = next(model.parameters()).device
    source, target = data['source'].to(device), data['target'].to(device)
    t = torch.tensor([data['time']], dtype=torch.float32, device=device)
    fraction = t[:, None, None, None, None]
    state = (1-fraction)*source + fraction*target
    opts[0].zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=cfg['amp'] and device.type == 'cuda'):
        raw_velocity = model(state, t, source)
    velocity = raw_velocity.float()
    fm = F.mse_loss(velocity, target-source)
    endpoint = state + (1-fraction)*velocity
    l1, roi = endpoint_l1(endpoint, target, cfg)
    proj, valid = endpoint.new_zeros(()), 0
    projection_active = cfg['projection_mode'] != 'none' and data['time'] == 0.
    if projection_active:
        proj, _, _, valid = projection_loss(endpoint, target, data['layers'])
    loss = fm + cfg['lambda_projection']*proj + cfg['lambda_endpoint_l1']*l1
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite loss; no skipped updates')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
    if not np.isfinite(float(norm)) or float(norm) <= 0:
        raise FloatingPointError('Nonfinite/zero total gradient')
    opts[0].step()
    return dict(loss=float(loss.detach()), fm_mse=float(fm.detach()),
        endpoint_l1=float(l1.detach()), weighted_endpoint_l1=float(cfg['lambda_endpoint_l1']*l1.detach()),
        projection_l1=float(proj.detach()), projection_active=float(projection_active),
        valid_slab_rays=float(valid), time=data['time'], gradient_norm=float(norm),
        exact_start=float(data['time'] == 0.), forward_calls=1.,
        model_output_bfloat16=float(raw_velocity.dtype == torch.bfloat16),
        step_seconds=time.perf_counter()-tick, **roi)


def predict(model, source, cfg):
    return generate_volume(model, source, cfg['patch'], steps=cfg['ode_steps'], overlap=cfg['overlap'], amp=cfg['amp'])

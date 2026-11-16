"""Paired 3D flow matching and t=0 projections from the same signed endpoint."""

import math

import numpy as np
import torch
from torch.nn import functional as F

from fm3d.inference import generate_volume
from fm3d.model import make_model as parent_model


PROJECTION_KEYS = ('projection_max_ilm_opl', 'projection_mean_ilm_opl',
                   'projection_mean_opl_bm', 'projection_mean_ilm_bm')


def make_model(cfg):
    return parent_model(cfg['base_channels'], condition=True, representation='voxel')


def optimizers(model, cfg):
    return [torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])]


def batch(inputs, row, cfg, time):
    source, target, origin, slices = inputs.training_patch(row)
    layers = inputs.layers(row['case_id'])[:, slices[1], slices[2]]
    return dict(source=source, target=target, layers=layers, time=float(time), origin_ZYX=origin)


def slab_masks(layers, depth, device):
    layers = torch.as_tensor(np.array(layers, copy=True), dtype=torch.float32, device=device)
    if (layers.ndim != 3 or layers.shape[0] != 6 or not torch.isfinite(layers).all()
            or layers.min() < 1 or layers.max() > 640 or (layers[1:] < layers[:-1]).any()):
        raise ValueError('Expected ordered MATLAB-one-based native640 layer coordinates [6,Y,X]')
    centers = ((torch.arange(depth, device=device, dtype=torch.float32) + .5) * 640 / depth - .5)[:, None, None]
    ilm, opl, bm = layers[0] - 1, layers[2] - 1, layers[5] - 1
    return dict(max_ilm_opl=(centers >= ilm) & (centers <= opl),
                mean_ilm_opl=(centers >= ilm) & (centers < opl),
                mean_opl_bm=(centers >= opl) & (centers <= bm),
                mean_ilm_bm=(centers >= ilm) & (centers <= bm))


def projected_l1(endpoint, target, layers, enabled):
    """One L1 per object, averaged over valid lateral rays, retaining signed values."""
    if endpoint.shape != target.shape or endpoint.ndim != 5 or endpoint.shape[:2] != (1, 1):
        raise ValueError('Expected aligned single-channel batch1 [1,1,Z,Y,X]')
    estimate, truth = endpoint[0, 0], target[0, 0]
    masks = slab_masks(layers, estimate.shape[0], estimate.device)
    values, counts = {}, {}
    for key in PROJECTION_KEYS:
        mask = masks[key.removeprefix('projection_')]
        count = mask.sum(0)
        valid = count > 0
        counts[key] = int(valid.sum())
        if not enabled[key] or not valid.any():
            values[key] = estimate.sum() * 0.
            continue
        if key == 'projection_max_ilm_opl':
            pred = estimate.masked_fill(~mask, -torch.inf).max(0).values
            gt = truth.masked_fill(~mask, -torch.inf).max(0).values
        else:
            pred = estimate.masked_fill(~mask, 0).sum(0) / (count + 1e-6)
            gt = truth.masked_fill(~mask, 0).sum(0) / (count + 1e-6)
        values[key] = (pred[valid] - gt[valid]).abs().mean()
    return values, counts


def loss_terms(model, data, cfg):
    device = next(model.parameters()).device
    source, target = data['source'].to(device), data['target'].to(device)
    time = float(data['time'])
    if not 0 <= time < 1:
        raise ValueError('Training samples require 0 <= t < 1')
    t = source.new_tensor([time])
    tau = t[:, None, None, None, None]
    state = (1-tau)*source + tau*target
    velocity = model(state, t, source)
    if velocity.dtype != torch.float32:
        raise TypeError('This experiment uses float32 tensors without autocast')
    terms = dict(fm=F.mse_loss(velocity, target - source))
    counts = {key: 0 for key in PROJECTION_KEYS}
    if time == 0.:
        endpoint = state + (1-tau)*velocity
        projection, counts = projected_l1(endpoint, target, data['layers'],
                                         {key: cfg['loss_weights'][key] > 0 for key in PROJECTION_KEYS})
        terms.update(projection)
    else:
        terms.update({key: velocity.sum() * 0. for key in PROJECTION_KEYS})
    total = sum(cfg['loss_weights'][key] * value for key, value in terms.items())
    return total, terms, counts


def update(model, opts, data, cfg):
    opts[0].zero_grad(set_to_none=True)
    total, terms, counts = loss_terms(model, data, cfg)
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite loss')
    total.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True))
    if not math.isfinite(norm) or norm <= 0:
        raise FloatingPointError('Nonfinite or zero gradient')
    opts[0].step()
    values = dict(loss=float(total.detach()), gradient_norm=norm, time=data['time'],
                  exact_start=float(data['time'] == 0.))
    for key, value in terms.items():
        values[key] = float(value.detach())
        values['weighted_' + key] = cfg['loss_weights'][key] * values[key]
    values.update({key + '_valid_rays': float(count) for key, count in counts.items()})
    return values


def gradient_diagnostics(model, data, cfg):
    """Measure weighted gradients on training data without choosing or changing weights."""
    _, terms, counts = loss_terms(model, data, cfg)
    parameters = tuple(model.parameters())
    result = {}
    for key, value in terms.items():
        if cfg['loss_weights'][key] == 0:
            result[key] = dict(raw=float(value.detach()), weight=0., gradient_l2=0.)
            continue
        gradients = torch.autograd.grad(value * cfg['loss_weights'][key], parameters,
                                        retain_graph=True, allow_unused=True)
        norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach())
                                                    for g in gradients if g is not None]))
        result[key] = dict(raw=float(value.detach()), weight=cfg['loss_weights'][key], gradient_l2=float(norm))
    return dict(terms=result, valid_rays=counts, weights_changed=False)


def predict(model, source, cfg):
    return generate_volume(model, source, cfg['patch'], steps=cfg['ode_steps'],
                           overlap=cfg['overlap'], amp=False)

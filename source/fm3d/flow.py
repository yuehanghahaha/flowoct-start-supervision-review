"""RMS-normalized conditional flow with analytic Gaussian reference velocity."""
import torch
from torch.nn import functional as F
from .wavelet import haar, inverse_haar


def slab_max(volume, layers, lower, upper, native_depth):
    depth = volume.shape[2]
    centers = (torch.arange(depth, device=volume.device, dtype=torch.float32) + .5) * (native_depth / depth) - .5
    z = centers[None, :, None, None]
    mask = (z >= layers[:, lower:lower + 1] - 1) & (z <= layers[:, upper:upper + 1] - 1)
    valid = mask.any(dim=1)
    maxima = volume[:, 0].masked_fill(~mask, torch.finfo(volume.dtype).min).amax(dim=1)
    return torch.where(valid, maxima, 0.0), valid


def reference_velocity(state, time):
    t = time[:, None, None, None, None].float()
    q = (1 - t).square() + t.square()
    return (2 * t - 1) / q * state.float(), q


def residual_velocity(model, state, features, time):
    reference, q = reference_velocity(state, time)
    with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16, enabled=state.is_cuda):
        residual = model(state / q.sqrt(), features, time)
    return residual.float(), reference


def encode(model, source):
    bands = haar(source.float())
    with torch.autocast(device_type=source.device.type, dtype=torch.bfloat16, enabled=source.is_cuda):
        return model.encode_oct(bands)


def objective(model, source, target, layers, noise, time, core, loss_config, native_depth, auxiliary_factor):
    # Haar, trajectory, residual target and all loss reductions remain float32.
    target_bands = haar(target.float()) / model.band_scale
    t = time[:, None, None, None, None].float()
    state = (1 - t) * noise + t * target_bands
    wanted = target_bands - noise
    residual, reference = residual_velocity(model, state, encode(model, source), time)
    residual_target = wanted - reference
    low = (residual[:, :1] - residual_target[:, :1]).square().flatten(1).mean(1)
    high = (residual[:, 1:] - residual_target[:, 1:]).square().flatten(1).mean(1)
    velocity = loss_config['velocity_low'] * low + loss_config['velocity_detail'] * high
    # Every sampled time receives endpoint supervision. No clipping during training.
    endpoint = inverse_haar((state + (1 - t) * (reference + residual)) * model.band_scale)
    image = (endpoint - target).abs().flatten(1).mean(1) / model.loss_scale[0]
    projections = []
    for i, (lower, upper) in enumerate(((0, 2), (2, 5)), start=1):
        prediction, valid = slab_max(endpoint, layers, lower, upper, native_depth)
        truth, _ = slab_max(target, layers, lower, upper, native_depth)
        projections.append(((prediction - truth).abs() * valid).flatten(1).sum(1) /
                           valid.flatten(1).sum(1).clamp_min(1) / model.loss_scale[i])
    mix = loss_config['mip_shallow_fraction']
    total = velocity + auxiliary_factor * (loss_config['endpoint_mae'] * image + loss_config['endpoint_mip'] *
                                           (mix * projections[0] + (1 - mix) * projections[1]))
    components = dict(loss=total.mean(), velocity_low=low.mean(), velocity_detail=high.mean(),
                      velocity=velocity.mean(), endpoint_mae=image.mean(),
                      endpoint_ilm_opl_mip=projections[0].mean(), endpoint_opl_bm_mip=projections[1].mean())
    return total.mean(), components


def heun_step(velocity, state, time, dt):
    first = velocity(state, time)
    second = velocity(state + dt * first, time + dt)
    return state + .5 * dt * (first + second)


@torch.inference_mode()
def generate(model, source, noise, steps, core_yx=None):
    features = encode(model, source)
    state = noise.float()
    def velocity(current, time):
        times = current.new_full((current.shape[0],), time)
        residual, reference = residual_velocity(model, current, features, times)
        return reference + residual
    for i in range(steps):
        state = heun_step(velocity, state, i / steps, 1 / steps)
        if not torch.isfinite(state).all().item():
            raise FloatingPointError(f'Nonfinite ODE state at Heun step {i + 1}.')
    return inverse_haar(state * model.band_scale)

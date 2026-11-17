"""Sliding full-volume velocity and unclamped integration on the parent grid."""
from itertools import product

import torch

from fm3d.inference import window_starts


class SlidingVelocity:
    def __init__(self, model, source, patch, overlap, merge_device=None):
        self.model = model
        self.device = next(model.parameters()).device
        self.merge_device = torch.device(merge_device) if merge_device else self.device
        self.source = source.to(self.merge_device).contiguous()
        starts = [window_starts(n, width, overlap) for n, width in zip(source.shape[2:], patch)]
        self.regions = [(slice(None), slice(None), *(slice(s, s+p) for s, p in zip(pos, patch)))
                        for pos in product(*starts)]
        axes = [torch.hann_window(n, periodic=False).clamp_min(.05) for n in patch]
        self.weight = (axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :])[None, None]
        self.weight = self.weight.to(self.merge_device)
        self.count = torch.zeros_like(self.source)
        for region in self.regions:
            self.count[region] += self.weight
        if self.count.min() <= 0:
            raise ValueError('Sliding windows must cover the full grid')
        self.calls = 0

    def __call__(self, state, t):
        velocity = torch.zeros_like(self.source)
        time = torch.full((1,), t, dtype=torch.float32, device=self.device)
        # Keep one window per forward and the parent's ordered Hann accumulation.
        condition = self.source
        for region in self.regions:
            outcome = self.model(state[region].to(self.device), time, condition[region].to(self.device))
            if outcome.dtype != torch.float32:
                raise TypeError('Expected float32 velocity with autocast disabled')
            velocity[region] += outcome.to(self.merge_device) * self.weight
        velocity.div_(self.count)
        if not torch.isfinite(velocity).all():
            raise FloatingPointError('Nonfinite stitched velocity')
        self.calls += 1
        return velocity


def interpolate(source, target, t):
    return (1-t)*source + t*target


def advance(velocity, start, steps, solver='Euler', first_step=0, first_velocity=None):
    """Integrate from first_step/steps to 1, with a fixed step size and no clamping."""
    state = start
    dt = 1 / steps
    for step in range(first_step, steps):
        v1 = first_velocity if step == first_step and first_velocity is not None else velocity(state, step*dt)
        if solver == 'Heun':
            predictor = state.add(v1, alpha=dt)
            v2 = velocity(predictor, (step+1)*dt)
            state = state.add((v1 + v2)*.5, alpha=dt)
        elif solver == 'Euler':
            state = state.add(v1, alpha=dt)
        else:
            raise ValueError(f'Unknown solver: {solver}')
    return state


def errors(estimate, truth):
    difference = estimate - truth
    return dict(mae=float(difference.abs().mean(dtype=torch.float64)),
                bias=float(difference.mean(dtype=torch.float64)),
                mse=float(difference.square().mean(dtype=torch.float64)))


def diagnostics(velocity, source, target):
    """Yield paired states and statistics at identical times, reusing the t=0 forward."""
    state = source
    true_velocity = target - source
    dt = 1 / 8
    for step in range(9):
        t = step*dt
        ideal = interpolate(source, target, t)
        deviation = state - ideal
        track = dict(step=step, t=t, **errors(state, ideal),
                     negative_ratio=float((state < 0).float().mean()),
                     overflow_ratio=float((state > 1).float().mean()),
                     clipped=errors(state.clamp(0, 1), ideal))
        velocity_record = None
        next_state = None
        if step < 8:
            free = velocity(state, t)
            teacher = free if step == 0 else velocity(ideal, t)
            free_error, teacher_error = free - true_velocity, teacher - true_velocity
            teacher_mse = float(teacher_error.square().mean(dtype=torch.float64))
            free_mse = float(free_error.square().mean(dtype=torch.float64))
            inner = float((deviation * free_error).mean(dtype=torch.float64))
            next_state = state.add(free, alpha=dt)
            next_ideal = interpolate(source, target, t+dt)
            next_mse = errors(next_state, next_ideal)['mse']
            predicted_change = 2*dt*inner + dt**2*free_mse
            velocity_record = dict(step=step, t=t, teacher_mse=teacher_mse, free_mse=free_mse,
                                   teacher_free_gap=free_mse-teacher_mse,
                                   difference_mse=float((free-teacher).square().mean(dtype=torch.float64)),
                                   teacher_velocity_bias=float(teacher_error.mean(dtype=torch.float64)),
                                   free_velocity_bias=float(free_error.mean(dtype=torch.float64)),
                                   error_velocity_inner=inner, next_state_mse=next_mse,
                                   change_in_state_mse=next_mse-track['mse'], predicted_mse_change=predicted_change,
                                   mse_balance_residual=next_mse-track['mse']-predicted_change)
        yield track, velocity_record, state, ideal
        state = next_state


def perturbations(ideal, small_mask, seed, cfg):
    for offset in cfg['offsets']:
        yield f'offset_{offset:+g}', ideal + offset
    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(ideal.shape, generator=generator, dtype=torch.float32).to(ideal.device)
    yield 'gaussian', ideal + cfg['noise_std']*noise
    if small_mask.any():
        mask = torch.from_numpy(small_mask)[None, None].to(ideal.device)
        yield 'small_vessel_attenuation', torch.where(mask, ideal*(1-cfg['small_vessel_attenuation']), ideal)


def xoct_endpoint(model, source):
    """The original XOCT 4x4 96-pixel tile grid and uniform averaging."""
    device = next(model.parameters()).device
    source = source.to(device).mul(2).sub(1)
    # The generator returns its 3D output before layer-projection branches.
    # These downstream branches never feed back into the requested volume.
    starts = [torch.linspace(0, n-96, 4).long().tolist() for n in source.shape[-2:]]
    output, count = torch.zeros_like(source), torch.zeros_like(source)
    for y, x in product(*starts):
        region = (..., slice(y, y+96), slice(x, x+96))
        prediction, _ = model.generator_3d(source[region])
        output[region] += prediction
        count[region] += 1
    return (output/(count+1e-6)+1)*.5

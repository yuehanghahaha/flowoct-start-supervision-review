"""Synthetic equations and original-inference parity on the first validation case."""
import argparse
import json
import time

import numpy as np
import torch

from fm3d.inference import generate_volume
from measure import Views
from metrics import quality as parent_quality
from shared import ROOT, config, configure_gpu, elapsed, inputs, model_weights, now, read, runtime, sha, write
from trajectory import SlidingVelocity, advance, diagnostics, interpolate


def synthetic():
    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_parameter('anchor', torch.nn.Parameter(torch.zeros(())))

        def forward(self, state, t, condition):
            return torch.full_like(state, .25)

    source = torch.full((1, 1, 16, 15, 14), .875)
    truth = source + .25
    velocity = SlidingVelocity(Constant().eval(), source, (16, 8, 8), .5)
    trajectory = list(diagnostics(velocity, source, truth))
    assert velocity.calls == 15 and len(trajectory) == 9
    for track, v, state, expected in trajectory:
        torch.testing.assert_close(state, expected, atol=2e-6, rtol=0)
        if v:
            assert abs(v['mse_balance_residual']) < 2e-8
    assert trajectory[-1][0]['overflow_ratio'] == 1.
    for solver in ('Euler', 'Heun'):
        for steps in (1, 2, 8, 32):
            torch.testing.assert_close(advance(velocity, source, steps, solver), truth, atol=2e-6, rtol=0)
    ideal = interpolate(source, truth, .5)
    torch.testing.assert_close(advance(velocity, ideal+.02, 8, first_step=4), truth+.02, atol=2e-6, rtol=0)
    # Test a time-dependent slope with a known integration result.
    def time_velocity(state, t):
        return torch.full_like(state, t)
    torch.testing.assert_close(advance(time_velocity, source, 8, 'Heun'), source+.5)
    torch.testing.assert_close(advance(time_velocity, source, 8), source+(7/16))
    receipt = dict(passed=True, time=now(), checks=['positive Hann coverage', 'linear trajectory and paired velocity',
                   'state error balance', 'unclamped states', 'Euler and Heun', 'time-dependent Heun',
                   'perturbation start and matched control'])
    write(ROOT/'verification/synthetic.json', receipt)
    print('Synthetic trajectory, integration and perturbation checks passed', flush=True)


@torch.inference_mode()
def a800():
    cfg = config()
    frozen_path = ROOT/'verification/frozen.json'
    frozen = json.loads(read(frozen_path))
    device = configure_gpu()
    cid = cfg['first_case']
    source, truth, layers, labels = inputs(cid, cfg, frozen)
    model, digest = model_weights('E090', cfg, frozen)
    model = model.to(device)
    velocity = SlidingVelocity(model, source, cfg['patch'], cfg['overlap'])
    tick = time.perf_counter()
    terminal = advance(velocity, velocity.source, 8).clamp(0, 1).cpu()
    gpu_time = elapsed(tick)
    tick = time.perf_counter()
    parent = generate_volume(model, source, cfg['patch'], steps=8, overlap=cfg['overlap'], amp=False)
    cpu_time = elapsed(tick)
    torch.testing.assert_close(terminal, parent, atol=2e-5, rtol=2e-4)
    array = terminal[0, 0].numpy()
    gt = truth[0, 0].numpy()
    views = Views(layers, gt.shape)
    legacy, projection, truth_projection, _ = parent_quality(array, gt, layers, {}, include_reference=False)
    new = views(array)
    for name, projected in projection.items():
        np.testing.assert_array_equal(new[name], projected)
    receipt = dict(passed=True, time=now(), case_id=cid, split='validation', tensor_sha256=digest,
                   frozen_sha256=sha(frozen_path), windows=len(velocity.regions),
                   max_abs_difference=float((terminal-parent).abs().max()),
                   mean_abs_difference=float((terminal-parent).abs().mean()), original_cpu_merge_seconds=cpu_time,
                   gpu_merge_seconds=gpu_time, shape=list(array.shape), projected_maps_identical=list(new),
                   gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), environment=runtime(),
                   same_case_parent_quality=legacy)
    write(ROOT/'verification/a800_preflight.json', receipt)
    print(f'A800 full-resolution parity passed, 8-step inference {gpu_time:.2f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('check', choices=['synthetic', 'a800'])
    {'synthetic': synthetic, 'a800': a800}[parser.parse_args().check]()

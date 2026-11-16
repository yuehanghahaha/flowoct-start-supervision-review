"""CPU contract checks and speed/whole-volume checks on one training patient."""
import argparse
import json
import statistics
import time

import numpy as np
import torch

import common as c
from fm3d.data import load_pair
from fm3d.inference import generate_volume
from fm3d.model import VelocityUNet3D
from metrics import PROTOCOL, quality
import method


def save(name, **evidence):
    result = dict(passed=True, source_contract=c.source_contract(), **evidence)
    c.write(c.VERIFY / name, result)
    print(json.dumps(evidence), flush=True)


def synthetic(cfg):
    c.configure(cfg, device='cpu')
    c.seed_all(cfg['seed'])
    parent = VelocityUNet3D(cfg['base_channels'], condition=True)
    c.seed_all(cfg['seed'])
    model = method.make_model(cfg)
    for name, value in parent.state_dict().items():
        reference = value[:, :1] if name == 'stem.weight' else value
        torch.testing.assert_close(model.state_dict()[name], reference, rtol=0, atol=0)
    source = torch.rand(1, 1, 16, 16, 16)
    target = torch.rand_like(source)
    t = torch.tensor([0.37])
    with torch.no_grad():
        velocity, loss = method.velocity_and_loss(model, source, target, t, False)
        expected = parent((1-t)*source+t*target, t, torch.zeros_like(source))
        torch.testing.assert_close(velocity, expected, rtol=2e-5, atol=3e-6)
        torch.testing.assert_close(loss, (velocity-(target-source)).square().mean(), rtol=1e-6, atol=1e-8)
    try:
        model(source, t, source)
    except ValueError:
        pass
    else:
        raise AssertionError('Explicit OCT conditioning accepted by the training model')
    before = c.model_digest(model)
    update = method.update(model, method.make_optimizer(model, cfg), source, target, float(t[0]), cfg)
    assert c.model_digest(model) != before

    class ToyField(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.calls = []

        def forward(self, x, time):
            self.calls.append((float(time[0]), float(x.mean())))
            return 0.3*x + time[0]*0.1

    toy = ToyField()
    oct_volume = torch.full((1, 1, 8, 13, 15), .22)
    actual = generate_volume(toy, oct_volume, (8, 8, 8), 8)
    expected = 0.22
    for step in range(8):
        expected += (0.3*expected + step/8*.1)/8
    torch.testing.assert_close(actual, torch.full_like(actual, expected), rtol=1e-6, atol=1e-7)
    assert abs(toy.calls[0][1]-.22) < 1e-6 and toy.calls[0][0] == 0.
    assert len({t for t, _ in toy.calls}) == 8 and toy.training
    for epoch in range(1, cfg['epochs']+1):
        times = c.time_schedule(cfg, epoch)
        assert len(times) == cfg['steps_per_epoch'] and np.all((0 < times) & (times < 1))
    save('synthetic.json', stem_input_channels=model.stem.in_channels,
         model_parameters=sum(p.numel() for p in model.parameters()),
         removed_parameters=parent.stem.weight.numel()-model.stem.weight.numel(),
         same_fresh_parent_weights_except_removed_channel=True,
         single_input_forward_and_pure_mse=True, optimizer_step=update, global_euler8=True,
         uniform_time_all_epochs=True)


def training_case(cfg):
    deployment, rows = c.binding()
    row = next(row for row in rows if row['split'] == 'train')
    c.configure(cfg)
    c.seed_all(cfg['seed'])
    return c.Inputs(deployment, cfg), row


def benchmark(cfg):
    inputs, row = training_case(cfg)
    source, target, origin = inputs.training_patch(row)
    source, target = source.to(cfg['device']), target.to(cfg['device'])
    tick = torch.tensor([0.37], dtype=torch.float32, device=cfg['device'])
    variants = {}
    with torch.no_grad():
        for label, amp in [('tf32_allowed', False), ('bf16_autocast', True)]:
            c.seed_all(cfg['seed'])
            model = method.make_model(cfg).to(cfg['device'])
            velocity, loss = method.velocity_and_loss(model, source, target, tick, amp)
            assert velocity.dtype == (torch.bfloat16 if amp else torch.float32)
            assert torch.isfinite(velocity).all() and torch.isfinite(loss) and loss.dtype == torch.float32
            variants[label] = dict(model=model, optimizer=method.make_optimizer(model, cfg),
                                   config=dict(cfg, amp=amp), initial_loss=float(loss), velocity=velocity.float(),
                                   times=[], first_digest=c.model_digest(model))
    base = variants['tf32_allowed']
    mixed = variants['bf16_autocast']
    relative = float((mixed['velocity']-base['velocity']).norm()/base['velocity'].norm())
    for variant in variants.values():
        del variant['velocity']
    results = {}
    # Alternate the order and repeat on the same crop to limit shared-GPU timing bias.
    warmup, measured = 4, 12
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(warmup+measured):
        items = list(variants.items())
        for label, variant in (items if iteration % 2 == 0 else reversed(items)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            values = method.update(variant['model'], variant['optimizer'], source, target, 0.37, variant['config'])
            torch.cuda.synchronize()
            if iteration >= warmup:
                variant['times'].append(time.perf_counter()-start)
            results[label] = values
    profiles = {}
    for label, variant in variants.items():
        assert c.model_digest(variant['model']) != variant['first_digest']
        profiles[label] = dict(median_step_seconds=statistics.median(variant['times']),
                               step_seconds=variant['times'], first_mse=variant['initial_loss'],
                               last=results[label])
    save('precision_benchmark.json', case_id=row['case_id'], split='train', origin_ZYX=list(origin),
         patch=cfg['patch'], warmup_updates_per_profile=warmup, timed_updates_per_profile=measured,
         timing_scope='forward/backward/optimizer, inputs already on GPU; shared card',
         gpu=torch.cuda.get_device_name(), precision=c.precision_flags(),
         profiles=profiles, initial_velocity_relative_l2_bf16_vs_tf32=relative,
         speedup=profiles['tf32_allowed']['median_step_seconds']/profiles['bf16_autocast']['median_step_seconds'],
         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)


def full(cfg):
    inputs, row = training_case(cfg)
    source, target = inputs.pair(row)
    reference = load_pair(row['path'], 'ZYX', 255., 256)
    assert np.array_equal(source, reference[0].numpy()) and np.array_equal(target, reference[1].numpy())
    del reference
    model = method.make_model(cfg).to(cfg['device'])
    start = time.perf_counter()
    prediction = method.predict(model, torch.from_numpy(np.array(source, copy=True))[None], cfg)
    seconds = time.perf_counter()-start
    assert prediction.shape == (1, 1, 256, 304, 304) and prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1
    scores, projections, _, slices = quality(prediction[0, 0].numpy(), target[0],
                                               inputs.layers(row['case_id']), {}, include_reference=False)
    objects = ['volume', 'bscan2d'] + ['projection_' + name.lower() for name in projections]
    assert len(projections) == 7 and len(slices) == 304
    assert all(np.isfinite(scores[f'{name}_{key}']) for name in objects for key in ('mae', 'psnr', 'ssim'))
    save('full_grid.json', case_id=row['case_id'], split='train', shape=list(prediction.shape),
         canonical_npz_and_mmap_cache_exactly_equal=True, inference_seconds=seconds,
         main_and_supplementary_objects=objects, metrics=scores, protocol=PROTOCOL,
         input_identities=inputs.identities, full_3d_files_saved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['synthetic', 'benchmark', 'full'])
    action = parser.parse_args().action
    {'synthetic': synthetic, 'benchmark': benchmark, 'full': full}[action](c.read(c.ROOT / 'experiment.json'))

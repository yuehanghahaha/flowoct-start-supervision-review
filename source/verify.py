"""Scientific invariants and restart/accumulation checks on synthetic CPU data.

The real experiment is accepted separately on full-resolution training patients.
No validation or test images are opened by this program.
"""
from copy import deepcopy
import hashlib
import json
import torch
from dataset import schedule, Inputs
from fm3d.model import WaveletVelocity, AxialBlock
from fm3d.wavelet import haar, inverse_haar
from fm3d.flow import reference_velocity, objective
from runtime import ROOT, load_config, load_checkpoint, save_checkpoint, write_json, now
from train import make_optimizer, snapshot, update_ema, learning_rate
from verification.reference_scan import ReferenceMamba


def main():
    torch.set_num_threads(4)
    torch.manual_seed(172)
    config = load_config()
    rows = [dict(case_id=str(i), split='train') for i in range(140)]
    batches = list(schedule(config, 1, rows))
    assert len(batches) == 200
    flat = [t for _, times in batches for t in times]
    assert flat.count(0.) == 100 and all(0 <= t < 1 for t in flat)
    visits = [r['case_id'] for rs, _ in batches for r in rs]
    assert sorted(visits.count(str(i)) for i in range(140)) == [2]*20 + [3]*120
    assert all(rs[0]['case_id'] != rs[1]['case_id'] for rs, _ in batches)
    assert list(schedule(config, 1, rows))[73:] == batches[73:]
    guard = object.__new__(Inputs); guard.allowed_splits = {'train'}
    try: guard.split('test')
    except ValueError: pass
    else: raise AssertionError('Test access guard failed.')

    volume = torch.randn(2, 1, 16, 16, 16)
    bands = haar(volume)
    torch.testing.assert_close(inverse_haar(bands), volume, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(bands.square().sum(), volume.square().sum(), rtol=1e-6, atol=1e-3)
    z0, z1 = torch.randn(2, 8, 3, 4, 5), torch.randn(2, 8, 3, 4, 5)
    for t in (0., .001, .125, .5, .875, .99999, 1.):
        time = torch.full((2,), t)
        state = (1-t)*z0 + t*z1
        reference, q = reference_velocity(state, time)
        residual = ((1-t)*z1 - t*z0) / q
        torch.testing.assert_close(reference + residual, z1-z0, atol=1e-6, rtol=2e-5)
        torch.testing.assert_close(state+(1-t)*(reference+residual), z1, atol=1e-6, rtol=2e-5)
        assert q.min() >= .5

    model = WaveletVelocity(config['model'], ReferenceMamba)
    counts = model.parameter_counts()
    assert counts['total'] == 38109512, counts
    # Nonzero diagnostic-only gates/head activate the complete backward graph.
    with torch.no_grad():
        model.out.weight.normal_(std=.01)
        for block in model.global_blocks:
            block.mod.bias[2*384:3*384] = .1
            block.mod.bias[5*384:6*384] = .1
    model.checkpoint_mode = 'none'
    source = torch.rand(2, 1, 16, 16, 16)
    target = torch.rand_like(source) * .2
    layers = torch.tensor([1., 90., 220., 350., 450., 640.])[None, :, None, None].expand(2, 6, 16, 16)
    noise = torch.randn(2, 8, 8, 8, 8)
    time = torch.tensor([0., .75])
    def gradient(begin, end, factor=1.):
        loss, _ = objective(model, source[begin:end], target[begin:end], layers[begin:end],
                            noise[begin:end], time[begin:end], None, config['loss'], 640, 1.)
        (factor*loss).backward()
        return float(loss)
    full_loss = gradient(0, 2)
    gradients = {n: p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    micro_loss = (gradient(0, 1, .5) + gradient(1, 2, .5)) * .5
    assert abs(full_loss-micro_loss) < 1e-5
    maximum = 0.
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.grad, gradients[name], atol=3e-6, rtol=3e-3)
        maximum = max(maximum, float((param.grad-gradients[name]).abs().max()))
    # Actual optimizer/RNG/EMA serialization and a subsequent update agree after restart.
    optimizer = make_optimizer(model, config)
    optimizer.step(); model.zero_grad(set_to_none=True)
    ema = deepcopy(model).requires_grad_(False)
    checkpoint = ROOT / 'verification/restart_check.pt'
    payload = snapshot(model, ema, optimizer, 0, config, torch.device('cpu'), {'test': True}, None, False)
    save_checkpoint(checkpoint, payload)
    def one_update():
        optimizer.zero_grad(set_to_none=True)
        loss = gradient(0, 2)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); update_ema(ema, model, config['ema_decay'])
        return loss
    one_update()
    expected = {n: p.detach().clone() for n, p in model.named_parameters()}
    expected_ema = {n: p.detach().clone() for n, p in ema.named_parameters()}
    payload = load_checkpoint(checkpoint)
    model.load_state_dict(payload['model']); ema.load_state_dict(payload['ema'])
    optimizer.load_state_dict(payload['optimizer']); torch.set_rng_state(payload['rng_cpu'])
    one_update()
    for n, p in model.named_parameters(): torch.testing.assert_close(p, expected[n], atol=0, rtol=0)
    for n, p in ema.named_parameters(): torch.testing.assert_close(p, expected_ema[n], atol=0, rtol=0)
    checkpoint.unlink()
    result = dict(state='passed', time=now(), parameters=counts, synthetic_only=True,
                  haar_roundtrip_energy=True, full_time_algebra=True, schedule_zero_fraction=.25,
                  patient_visits_per_epoch=400, no_test_access=True,
                  microbatch_gradient_max_absolute_error=maximum, optimizer_ema_resume_exact=True,
                  temporary_checkpoint_removed=True)
    write_json(ROOT / 'verification/correctness.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

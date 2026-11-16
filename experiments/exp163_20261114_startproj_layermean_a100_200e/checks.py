"""Independent loss/gradient regressions, resume checks and full-size hardware acceptance."""

import argparse
import copy
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scipy.io import loadmat
from skimage.metrics import structural_similarity

import common as c
import method
import metrics
import run


def layers_for(height, width):
    return np.broadcast_to(np.array([61, 101, 221, 301, 401, 541], dtype=np.float32)[:, None, None],
                           (6, height, width)).copy()


def assert_tree(left, right):
    if torch.is_tensor(left):
        assert torch.equal(left.cpu(), right.cpu())
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree(a, b)
    else:
        assert left == right


def mathematical_checks(cfg):
    results = []
    for t in (0., .23, .83, .999):
        source = torch.full((1, 1, 8, 9, 11), .15)
        target = torch.full_like(source, .65)
        velocity = (target-source).clone().requires_grad_(True)
        estimate = source.lerp(target, t) + (1-t)*velocity
        assert torch.allclose(estimate, target, atol=1e-6)
        gradient = torch.autograd.grad(estimate.sum(), velocity)[0]
        assert torch.allclose(gradient, torch.full_like(gradient, 1-t), atol=1e-6)
    results.append('Linear-path endpoint and its (1-t) gradient at four times')

    z, height, width = 16, 5, 7
    layers = layers_for(height, width)
    layers[:, 0, 0] = 1  # A legal empty lateral ray.
    target = torch.arange(z, dtype=torch.float32)[None, None, :, None, None].expand(1, 1, z, height, width)/100
    endpoint = (target-.4).clone().requires_grad_(True)  # Entirely negative, signed endpoint.
    values, counts = method.projected_l1(endpoint, target, layers, {key: True for key in method.PROJECTION_KEYS})
    centers = (np.arange(z, dtype=np.float32)+.5)*640/z-.5
    ilm, opl, bm = layers[0]-1, layers[2]-1, layers[5]-1
    masks = ((centers[:, None, None] >= ilm) & (centers[:, None, None] <= opl),
             (centers[:, None, None] >= ilm) & (centers[:, None, None] < opl),
             (centers[:, None, None] >= opl) & (centers[:, None, None] <= bm),
             (centers[:, None, None] >= ilm) & (centers[:, None, None] <= bm))
    for index, (key, mask) in enumerate(zip(method.PROJECTION_KEYS, masks)):
        truth = target[0, 0].numpy()
        pred = endpoint.detach()[0, 0].numpy()
        valid = mask.any(0)
        if index == 0:
            a = np.where(mask, pred, -np.inf).max(0)
            b = np.where(mask, truth, -np.inf).max(0)
        else:
            a = np.where(mask, pred, 0).sum(0)/(mask.sum(0)+1e-6)
            b = np.where(mask, truth, 0).sum(0)/(mask.sum(0)+1e-6)
        expected = np.abs(a[valid]-b[valid]).mean()
        assert np.isclose(float(values[key].detach()), expected, rtol=0, atol=1e-7)
        assert counts[key] == int(valid.sum()) == height*width-1
        gradient = torch.autograd.grad(values[key], endpoint, retain_graph=True)[0][0, 0].numpy()
        expected_grad = np.zeros_like(pred)
        if index == 0:
            argmax = np.where(mask, pred, -np.inf).argmax(0)
            for y, x in zip(*np.nonzero(valid)):
                expected_grad[argmax[y, x], y, x] = -1/valid.sum()
        else:
            expected_grad = np.where(mask, -1/(mask.sum(0)+1e-6)/valid.sum(), 0)
        assert np.allclose(gradient, expected_grad, rtol=0, atol=1e-7)
    means = [values[key] for key in method.PROJECTION_KEYS[1:]]
    full_gradient = torch.autograd.grad(sum(.1*value for value in means), endpoint, retain_graph=True)[0]
    average_gradient = torch.autograd.grad(sum(.1/3*value for value in means), endpoint)[0]
    assert torch.allclose(full_gradient, 3*average_gradient, rtol=1e-6, atol=1e-8)
    results.append('Four independent NumPy projections, analytic gradients, signed maximum and empty rays')
    results.append('Per-mean coefficient 0.1 is a sum, with exactly threefold projection gradient versus 0.1/3')

    empty = np.ones_like(layers)
    x = torch.randn(1, 1, z, height, width, requires_grad=True)
    errors, valid = method.projected_l1(x, target, empty, {key: True for key in method.PROJECTION_KEYS})
    sum(errors.values()).backward()
    assert torch.equal(x.grad, torch.zeros_like(x)) and not any(valid.values())
    results.append('Completely empty slabs produce finite zero losses and gradients')

    torch.manual_seed(11)
    net = method.make_model(dict(base_channels=4))
    sample = dict(source=torch.rand(1, 1, 8, 8, 8), target=torch.rand(1, 1, 8, 8, 8),
                  layers=layers_for(8, 8), time=.37)
    _, terms, _ = method.loss_terms(net, sample, cfg)
    assert all(float(terms[key]) == 0 for key in method.PROJECTION_KEYS)
    sample['time'] = 0.
    total, terms, _ = method.loss_terms(net, sample, cfg)
    expected = terms['fm'] + sum(cfg['loss_weights'][key]*terms[key] for key in method.PROJECTION_KEYS)
    assert torch.equal(total, expected)
    times, mask = c.time_schedule(cfg, 1)
    assert len(times) == 200 and (times == 0).sum() == mask.sum() == 100
    assert ((times[~mask] > 0) & (times[~mask] < 1)).all()
    results.append('Projection gate at exact t0, complete weighted formula, fixed 100+100 time schedule')

    control = copy.deepcopy(cfg)
    control['loss_weights'].update({key: 0. for key in method.PROJECTION_KEYS[1:]})
    actual, _, _ = method.loss_terms(net, sample, control)
    velocity = net(sample['source'], torch.zeros(1), sample['source'])
    original_projection = c.projection_loss(sample['source']+velocity, sample['target'], sample['layers'])[0]
    original = torch.nn.functional.mse_loss(velocity, sample['target']-sample['source'])+.1*original_projection
    assert torch.equal(actual, original)
    gradients = torch.autograd.grad(actual, tuple(net.parameters()))
    expected_gradients = torch.autograd.grad(original, tuple(net.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(gradients, expected_gradients))
    results.append('Control loss and all parameter gradients equal the retained exp018 formula exactly')

    rng = np.random.default_rng(5)
    a = rng.random((41, 11, 13), dtype=np.float32)
    b = (a*.7).astype(np.float32)
    direct = structural_similarity(a, b, data_range=1., win_size=7, gaussian_weights=False,
                                   use_sample_covariance=True, channel_axis=None)
    assert abs(metrics.window_ssim(b, a, 1.)-direct) < 2e-7
    results.append('Memory-bounded 3D SSIM agrees with direct full-neighborhood skimage scoring')

    guard = object.__new__(c.Inputs)
    guard.purpose = 'train'
    guard.splits = {'10305': 'test'}
    try:
        guard.pair(dict(case_id='10305'))
    except ValueError:
        pass
    else:
        raise AssertionError('Training accessed test input')
    results.append('Test case access rejected before any array or label file operation')
    return results


def resume_check(cfg):
    """Exercise the normal training/resume entry point with generated CPU fixtures."""
    settings = copy.deepcopy(cfg)
    settings.update(base_channels=4, device='cpu', epochs=2, steps_per_epoch=2, val_every=1, save_every=1,
                    patch=[8, 8, 8], threads=1)
    contract = {'generated_fixture': 'resume regression'}

    class Inputs:
        def __init__(self, *args):
            c.write(c.ROOT/'verification/input_identity.json', {})

        def training_patch(self, row):
            y = random.randint(0, 8)
            data = torch.linspace(.1, .8, 8**3).reshape(1, 1, 8, 8, 8)
            return data+y/100, data*.5, (0, y, 0), (slice(0, 8), slice(0, 8), slice(0, 8))

        def layers(self, cid):
            return layers_for(8, 8)

    rows = [dict(case_id='synthetic', patient_id='synthetic', split='train')]
    interrupted = [False]

    def evaluate(model, inputs, rows, config, split, epoch, digest):
        if interrupted[0]:
            interrupted[0] = False
            raise RuntimeError('simulated interruption during validation')
        # Increasing frozen fixture score makes the selected final checkpoint explicit.
        return dict(metrics={config['selection_metric']: epoch/10, 'reference_volume_psnr': 20+epoch}), 'generated_fixture'

    with tempfile.TemporaryDirectory(prefix='layerproj_resume_') as temporary:
        root = Path(temporary)
        manifest = root/'manifest.csv'
        manifest.write_text('generated fixture\n')
        deployment = dict(manifest=str(manifest))
        results = []
        for label in ('continuous', 'interrupted'):
            folder = root/label
            for name in ('verification', 'runs/train', 'checkpoints'):
                (folder/name).mkdir(parents=True)
            for name in ('preflight.json', 'full_preflight.json'):
                c.write(folder/'verification'/name, dict(passed=True, source_contract=contract))
            with ExitStack() as stack:
                for module in (c, run):
                    for attr, value in (('ROOT', folder), ('RUN', folder/'runs/train'), ('CKPT', folder/'checkpoints')):
                        stack.enter_context(patch.object(module, attr, value))
                stack.enter_context(patch.object(c, 'source_contract', lambda: contract))
                stack.enter_context(patch.object(c, 'Inputs', Inputs))
                stack.enter_context(patch.object(c, 'evaluate', evaluate))
                for name, function in (('reset_peak_memory_stats', lambda *a: None),
                                       ('max_memory_allocated', lambda *a: 0),
                                       ('max_memory_reserved', lambda *a: 0),
                                       ('get_device_name', lambda *a: 'generated CPU fixture')):
                    stack.enter_context(patch.object(torch.cuda, name, function))
                if label == 'interrupted':
                    interrupted[0] = True
                    try:
                        run.train(settings, deployment, rows, False)
                    except RuntimeError as error:
                        assert str(error) == 'simulated interruption during validation'
                    else:
                        raise AssertionError('Expected validation interruption')
                    assert (folder/'checkpoints/trained_pending_validation.pt').is_file()
                    run.train(settings, deployment, rows, True)
                else:
                    run.train(settings, deployment, rows, False)
                saved = torch.load(folder/'checkpoints/latest.pt', map_location='cpu', weights_only=False)
                complete = c.read(folder/'runs/train/training_complete.json')
                history = [json.loads(line) for line in (folder/'runs/train/history.jsonl').read_text().splitlines()]
                assert [item['epoch'] for item in history] == [1, 2]
                assert complete['updates'] == 4 and complete['best']['epoch'] == 2
                assert not (folder/'checkpoints/trained_pending_validation.pt').exists()
                selected = torch.load(folder/'checkpoints/best.pt', map_location='cpu', weights_only=False)
                assert selected['model_tensor_sha256'] == saved['model_tensor_sha256']
                results.append(saved)
        assert_tree(results[0]['model'], results[1]['model'])
        assert_tree(results[0]['optimizers'], results[1]['optimizers'])
        assert_tree(results[0]['rng_state'], results[1]['rng_state'])
    return 'Validation interruption/resume reproduces continuous model, AdamW moments, RNG, selection and history exactly'


def acceptance(cfg, real):
    c.configure(cfg)
    c.seed_all(cfg['seed'])
    device = torch.device(cfg['device'])
    if device.type != 'cuda':
        raise ValueError('GPU acceptance requires the configured CUDA device')
    torch.cuda.reset_peak_memory_stats(device)
    net = method.make_model(cfg).float().to(device)
    opts = method.optimizers(net, cfg)
    if real:
        args = SimpleNamespace(data_root=None, manifest=None, cache_root=None, labels_root=None)
        deployment, rows = c.binding(args)
        inputs = c.Inputs(deployment, cfg)
        patient = next(row for row in rows if row['split'] == 'train')
        data = method.batch(inputs, patient, cfg, 0.)
        case = patient['case_id']
    else:
        source = torch.rand(1, 1, *cfg['patch'])
        data = dict(source=source, target=source*.6, layers=layers_for(*cfg['patch'][1:]), time=0.)
        case = 'generated_fixture'
    before = c.model_digest(net)
    gradients = method.gradient_diagnostics(net, data, cfg)
    values = []
    for time_value in (0., .37):
        data['time'] = time_value
        values.append(method.update(net, opts, data, cfg))
    assert before != c.model_digest(net)
    assert all(np.isfinite(v) for row in values for v in row.values())
    with tempfile.TemporaryDirectory(prefix='layerproj_gpu_resume_') as temporary:
        path = Path(temporary)/'resume.pt'
        state = dict(model=copy.deepcopy(net.state_dict()), optimizers=copy.deepcopy([o.state_dict() for o in opts]),
                     rng_state=c.rng_state())
        c.save_checkpoint(path, state)
        loaded = torch.load(path, map_location='cpu', weights_only=False)
        assert_tree(state, loaded)
        run.restore_optimizers(opts, loaded['optimizers'], device)
        c.restore_rng(loaded['rng_state'])
    del data
    gc.collect()
    if real:
        source, target = inputs.pair(patient)
        full_source = torch.from_numpy(np.array(source, copy=True))[None]
    else:
        full_source = torch.full((1, 1, 256, 304, 304), .25)
    begin = time.perf_counter()
    prediction = method.predict(net, full_source, cfg)
    assert prediction.shape == (1, 1, 256, 304, 304) and prediction.dtype == torch.float32
    assert torch.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1
    inference_seconds = time.perf_counter()-begin
    metric_groups = None
    if real:
        scores, _, _, slices = metrics.quality(prediction[0, 0].numpy(), np.array(target[0]),
                            inputs.layers(case), inputs.regions_xy(case), include_reference=False)
        metric_groups = ['volume', 'bscan2d', 'projection_ilm_opl', 'projection_opl_bm', 'projection_ilm_bm_mean',
                         'projection_ilm_opl_mean', 'projection_opl_bm_mean', 'projection_full', 'projection_zmax']
        assert len(slices) == 304
        assert all(np.isfinite(scores[group+'_'+metric]) for group in metric_groups for metric in ('mae','psnr','ssim'))
    receipt = dict(passed=True, source_contract=c.source_contract(), versions=c.versions(),
                   precision_flags=c.precision_flags(net), case_id=case, source_split='train' if real else 'generated',
                   tensor_shape=list(prediction.shape), patch=cfg['patch'], updates=values, gradient_diagnostics=gradients,
                   full_euler8_seconds=inference_seconds, parameters=sum(p.numel() for p in net.parameters()),
                   gpu=torch.cuda.get_device_name(device), peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30, scored_groups=metric_groups,
                   validation_patients_used=0, test_patients_used=0, formal_training_started=False,
                   temporary_checkpoint_removed=True, formal_checkpoints_written=0)
    name = 'preflight.json' if real else 'gpu_synthetic.json'
    c.write(c.ROOT/'verification'/name, receipt)
    if real:
        c.write(c.ROOT/'verification/full_preflight.json', receipt)
    print(json.dumps({key:value for key,value in receipt.items() if key != 'source_contract'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('synthetic', 'gpu', 'real'))
    args = parser.parse_args()
    cfg = c.read(c.ROOT/'experiment.json')
    if args.action == 'synthetic':
        torch.set_num_threads(1)
        checks = mathematical_checks(cfg)
        checks.append(resume_check(cfg))
        c.write(c.ROOT/'verification/synthetic.json', dict(passed=True, checks=checks,
                source_contract=c.source_contract(), versions=c.versions(), test_patients_used=0,
                patient_arrays_used=0, temporary_fixtures_removed=True))
        print(f'Passed {len(checks)} scientific regression groups.', flush=True)
    else:
        acceptance(cfg, args.action == 'real')


if __name__ == '__main__':
    main()

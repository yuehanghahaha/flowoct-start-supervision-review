"""Run the two validation jobs and commit evidence per case or branch."""
import argparse
import fcntl
import json
import os
import time
import traceback

import numpy as np
import torch

from measure import (HISTOGRAM_EDGES, Views, corrections, distribution, intensity, plane_views,
                     quality, topology, vessel_regions)
from shared import (ROOT, commit, config, configure_gpu, elapsed, existing, inputs, model_weights,
                    now, read, runtime, save_array, save_views, sha, status, write)
from trajectory import (SlidingVelocity, advance, diagnostics, errors, interpolate, perturbations, xoct_endpoint)


def frozen_identity(frozen, tag, cid):
    return dict(model=frozen['models'][tag]['tensor_sha256'], input=frozen['inputs'][cid]['sha256'])


def measure_endpoint(raw, truth, views, gt_views, folder, retain_bscans=False):
    tick = time.perf_counter()
    array = raw.detach().cpu().numpy()[0, 0] if torch.is_tensor(raw) else raw
    clipped = array.clip(0, 1)
    metrics, projected, bscans = quality(clipped, truth, views, gt_views, retain_bscans)
    output = folder / 'view.npz'
    save_views(output, **plane_views(clipped, projected))
    err = array - truth
    metrics_raw = dict(mae=float(np.abs(err).mean(dtype=np.float64)),
                       bias=float(err.mean(dtype=np.float64)), mse=float(np.square(err).mean(dtype=np.float64)),
                       negative_ratio=float((array < 0).mean()), overflow_ratio=float((array > 1).mean()))
    return dict(standard=metrics, raw=metrics_raw, metric_seconds=elapsed(tick)), projected, bscans, output


def save_shared_truth(gt, gt_views, layers, masks, folder):
    path = folder/'view.npz'
    save_views(path, **plane_views(gt, gt_views), **{f'mask_{name}': value for name, value in masks.items()})
    volume = folder/'gt.npy'
    save_array(volume, gt)
    layer_file = folder/'layers.npy'
    save_array(layer_file, layers)
    return [path, volume, layer_file]


def core_case(tag, cid, model, x, y, layers, labels, cfg, frozen, device):
    tick = time.perf_counter()
    folder = ROOT/'reports/core'/tag/cid
    identity = frozen_identity(frozen, tag, cid)
    if existing(folder, identity):
        return
    gt = y[0, 0].numpy()
    views = Views(layers, gt.shape)
    gt_views = views(gt)
    regions, masks = vessel_regions(labels, gt, views, cfg['large_diameter_px'], labels.get('faz3d'))
    truth_folder = ROOT/'reports/truth'/cid
    if not existing(truth_folder, identity['input'], verify=False):
        truth_files = save_shared_truth(gt, gt_views, layers, masks, truth_folder)
        commit(truth_folder, dict(identity=identity['input'], case=cid, distribution=distribution(gt)), truth_files)
    x, y = x.to(device), y.to(device)
    evidence, track, velocities = [], [], []
    if tag == 'XOCT_E213':
        terminal = xoct_endpoint(model, x)
    else:
        velocity = SlidingVelocity(model, x, cfg['patch'], cfg['overlap'])
        trajectory_folder = ROOT/'predictions/trajectory'/tag/cid
        for state_row, velocity_row, state, ideal in diagnostics(velocity, x, y):
            step = state_row['step']
            raw = state[0, 0].cpu().numpy()
            target = ideal[0, 0].cpu().numpy()
            volume_path = trajectory_folder/f'step_{step:02d}.npy'
            save_array(volume_path, raw)
            state_view = plane_views(raw, views(raw))
            ideal_view = plane_views(target, views(target))
            pair_path = folder/'trajectory'/f'step_{step:02d}.npz'
            save_views(pair_path, **{f'free_{k}': v for k, v in state_view.items()},
                       **{f'ideal_{k}': v for k, v in ideal_view.items()})
            evidence += [volume_path, pair_path]
            track.append(state_row)
            if velocity_row is not None:
                velocities.append(velocity_row)
            terminal = state
            print(f'core {tag} {cid} trajectory {step}/8', flush=True)
        if velocity.calls != 15 or track[0]['mae'] != 0 or velocities[0]['teacher_free_gap'] != 0:
            raise ValueError('Unexpected trajectory start or velocity call count')
        if max(abs(row['mse_balance_residual']) for row in velocities) > 2e-8:
            raise ValueError('Euler error balance failed')
    endpoint, projected, bscans, path = measure_endpoint(terminal, gt, views, gt_views, folder, True)
    evidence.append(path)
    raw = terminal[0, 0].cpu().numpy()
    clipped = raw.clip(0, 1)
    distribution_after = intensity(clipped, gt, regions)
    topology_metrics = topology(projected['ILM_OPL'], gt_views['ILM_OPL'], masks)
    if tag == 'XOCT_E213':
        xoct_raw = folder/'raw.npy'
        save_array(xoct_raw, raw)
        evidence.append(xoct_raw)
    signed = {name: pred - plane_views(gt, gt_views)[name]
              for name, pred in plane_views(clipped, projected).items()}
    signed.update({f'raw_{name}': pred - plane_views(gt, gt_views)[name]
                   for name, pred in plane_views(raw, views(raw)).items()})
    error_path = folder/'signed_error.npz'
    save_views(error_path, **signed)
    evidence.append(error_path)
    bscan_path = folder/'bscans.json'
    write(bscan_path, bscans)
    evidence.append(bscan_path)
    oracle = {}
    for name, corrected in corrections(raw, gt):
        summary, _, _, path = measure_endpoint(corrected, gt, views, gt_views, folder/'oracle'/name)
        oracle[name] = summary
        evidence.append(path)
    commit(folder, dict(identity=identity, model=tag, case=cid, endpoint=endpoint,
                        raw=intensity(raw, gt, regions), clipped=distribution_after, topology=topology_metrics,
                        oracle=oracle, trajectory=track, velocity=velocities,
                        histogram_edges=HISTOGRAM_EDGES.tolist(), seconds=elapsed(tick)), evidence)
    print(f"core committed {tag} {cid} 3D SSIM={endpoint['standard']['volume']['ssim']:.6f}", flush=True)


def extended_case(tag, cid, model, x, y, layers, labels, cfg, frozen, device):
    folder = ROOT/'reports/extended'/tag/cid
    identity = frozen_identity(frozen, tag, cid)
    if existing(folder, identity):
        return
    gt = y[0, 0].numpy()
    views = Views(layers, gt.shape)
    gt_views = views(gt)
    regions, _ = vessel_regions(labels, gt, views, cfg['large_diameter_px'], labels.get('faz3d'))
    small = regions['small_vessel_candidate_3d']
    x, y = x.to(device), y.to(device)
    velocity = SlidingVelocity(model, x, cfg['patch'], cfg['overlap'])
    branch_records = []
    first_velocity = None
    for solver, step_values in cfg['solver_grid'].items():
        for steps in step_values:
            name = f'{solver}{steps}'
            branch = folder/'solver'/name
            branch_records.append(branch/'record.json')
            if existing(branch, identity):
                continue
            tick = time.perf_counter()
            nfe = steps*(2 if solver == 'Heun' else 1)
            core_folder = ROOT/'reports/core'/tag/cid
            core = existing(core_folder, identity, verify=False) if name == 'Euler8' else None
            provenance = 'fresh solver integration'
            before = velocity.calls
            if core:
                raw = np.load(ROOT/'predictions/trajectory'/tag/cid/'step_08.npy', mmap_mode='r', allow_pickle=False)
                provenance = 'same Euler8 endpoint from E/F'
            else:
                if first_velocity is None:
                    first_velocity = velocity(x, 0.0)
                raw = advance(velocity, x, steps, solver, first_velocity=first_velocity)
            result, _, _, path = measure_endpoint(raw, gt, views, gt_views, branch)
            commit(branch, dict(identity=identity, model=tag, case=cid, solver=solver, steps=steps, nfe=nfe,
                                physical_new_calls=velocity.calls-before, provenance=provenance,
                                seconds=elapsed(tick), **result), [path])
            print(f'extended committed {tag} {cid} {name}', flush=True)
    del first_velocity

    for start in cfg['perturb_steps']:
        ideal = interpolate(x, y, start/8)
        seed = cfg['noise_seed']+int(cid)*10+start
        subgroup = folder/'perturb'/f't{start:02d}'
        names = ['control', *[f'offset_{offset:+g}' for offset in cfg['offsets']], 'gaussian']
        if small.any():
            names.append('small_vessel_attenuation')
        group_records = {name: existing(subgroup/name, identity) for name in names}
        branch_records += [subgroup/name/'record.json' for name in names]
        if all(group_records.values()):
            continue
        tick = time.perf_counter()
        control = advance(velocity, ideal, 8, first_step=start)
        if group_records['control'] is None:
            branch = subgroup/'control'
            result, _, _, path = measure_endpoint(control, gt, views, gt_views, branch)
            commit(branch, dict(identity=identity, model=tag, case=cid, perturbation='control', start_step=start,
                                start_t=start/8, nfe=8-start, seconds=elapsed(tick), **result), [path])
        for name, perturbed in perturbations(ideal, small, seed, cfg):
            if group_records[name]:
                continue
            tick = time.perf_counter()
            branch = subgroup/name
            raw = advance(velocity, perturbed, 8, first_step=start)
            initial_diff, final_diff = errors(perturbed, ideal), errors(raw, control)
            clipped_diff = errors(raw.clamp(0, 1), control.clamp(0, 1))
            result, _, _, path = measure_endpoint(raw, gt, views, gt_views, branch)
            commit(branch, dict(identity=identity, model=tag, case=cid, perturbation=name, start_step=start,
                                start_t=start/8, seed=seed if name == 'gaussian' else None, nfe=8-start,
                                versus_control_initial=initial_diff, versus_control_final=final_diff,
                                versus_control_clipped=clipped_diff,
                                mae_ratio=final_diff['mae']/initial_diff['mae'] if initial_diff['mae'] else None,
                                signed_bias_ratio=final_diff['bias']/initial_diff['bias'] if name.startswith('offset_') else None,
                                seconds=elapsed(tick), **result), [path])
            print(f'extended committed {tag} {cid} t={start/8:g} {name}', flush=True)
        del control
    commit(folder, dict(identity=identity, model=tag, case=cid, branches=len(branch_records),
                        missing_G6=not bool(small.any())), branch_records)


@torch.inference_mode()
def run(phase, first=False):
    cfg = config()
    frozen = json.loads(read(ROOT/'verification/frozen.json'))
    check = json.loads(read(ROOT/'verification/a800_preflight.json'))
    if not check['passed'] or check['frozen_sha256'] != sha(ROOT/'verification/frozen.json'):
        raise ValueError('A800 validation preflight is missing for this frozen build')
    for relative, digest in frozen['code_sha256'].items():
        if sha(ROOT/relative) != digest:
            raise ValueError(f'Frozen experiment file changed: {relative}')
    device = configure_gpu()
    cases = [cfg['first_case']] if first else cfg['validation_cases']
    models = list(cfg['models']) if phase == 'core' else [tag for tag in cfg['models'] if tag != 'XOCT_E213']
    # Model order is fixed, with the first validation case across all weights first.
    loaded = {tag: model_weights(tag, cfg, frozen)[0].to(device) for tag in models}
    status(phase, 'running', completed=0, planned=len(models)*len(cases), environment=runtime())
    total = len(cases)*len(models)
    done = 0
    execute = core_case if phase == 'core' else extended_case
    for cid in cases:
        x, y, layers, labels = inputs(cid, cfg, frozen)
        for tag in models:
            status(phase, 'running', completed=done, planned=total, case=cid, model=tag, environment=runtime())
            print(f'{now()} {phase} {tag} {cid} ({done+1}/{total})', flush=True)
            execute(tag, cid, loaded[tag], x, y, layers, labels, cfg, frozen, device)
            done += 1
        del x, y, layers, labels
    status(phase, 'first_case_complete' if first else 'complete', completed=done, planned=total, environment=runtime())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['core', 'extended'])
    parser.add_argument('--first', action='store_true')
    args = parser.parse_args()
    directory = ROOT/'runs'
    directory.mkdir(exist_ok=True)
    with (directory/f'{args.phase}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args.phase, args.first)
        except Exception as exc:
            status(args.phase, 'failed', error=str(exc))
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()

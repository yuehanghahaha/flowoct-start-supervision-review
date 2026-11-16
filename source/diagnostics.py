"""Post-selection validation-only condition mismatch and fixed-time diagnostics."""
import numpy as np
import torch
from dataset import patient_noise, patient_seed
from fm3d.flow import encode, residual_velocity
from fm3d.wavelet import haar, inverse_haar
from metrics import macro, GROUPS
from report import generate_case, write_case, read_completed, write_tables
from runtime import now, write_json, read_json, pause_at_boundary


TIMES = (0., .125, .25, .5, .75, .875, .9375)


@torch.inference_mode()
def run_diagnostics(root, config, inputs, model, device, completion):
    rows = inputs.split('validation')
    if len(rows) != 10:
        raise ValueError('Diagnostics require only the fixed ten validation patients.')
    identity = completion['best_sha256']
    folder = root / 'runs/diagnostics'
    records = []
    for i, row in enumerate(rows):
        pause_at_boundary(root, 'diagnostics', patients=i, update=completion['updates'])
        donor = rows[(i + 1) % len(rows)]
        case_dir = folder / 'mismatched_oct' / row['case_id']
        receipt = dict(phase='validation_mismatched_oct_diagnostic', weight_identity=identity,
                       solver=config['solver'], inference_core=None, noise_seed=patient_seed(config, row),
                       image=inputs.identify(row, 'image'), layers=inputs.identify(row, 'layers'),
                       donor_case_id=donor['case_id'], donor_image=inputs.identify(donor, 'image'))
        record = read_completed(case_dir, receipt, False)
        probe_file = folder / 'time_probes' / (row['case_id'] + '.json')
        if record is not None and probe_file.is_file():
            previous = read_json(probe_file)
            if previous['weight_identity'] != identity or previous['donor_case_id'] != donor['case_id']:
                raise ValueError('Fixed-time diagnostic identity changed.')
            records.append(record)
            continue
        source = inputs.volume(row, 'oct')
        wrong_source = inputs.volume(donor, 'oct')
        target = inputs.volume(row, 'octa')
        layers = inputs.layers(row)
        noise, seed = patient_noise(config, row, (1, 8, 128, 152, 152))
        if record is None:
            prediction, seconds, clipped = generate_case(model, wrong_source, noise, config, device)
            record = write_case(case_dir, row['case_id'], prediction, target[0, 0].numpy(), layers,
                                dict(receipt, seconds=seconds, clipped_fraction=clipped), False,
                                wrong_source[0, 0].numpy())
            del prediction
        target = target.to(device)
        z1 = haar(target) / model.band_scale
        z0 = noise.to(device)
        features = [encode(model, source.to(device)), encode(model, wrong_source.to(device))]
        probes = []
        for t in TIMES:
            state = (1-t) * z0 + t * z1
            for label, feature in zip(('correct_oct', 'mismatched_oct'), features):
                residual, reference = residual_velocity(model, state, feature, state.new_tensor([t]))
                errors = (residual + reference - (z1-z0)).square().mean(dim=(0, 2, 3, 4)).sqrt()
                endpoint = inverse_haar((state + (1-t)*(reference+residual)) * model.band_scale)
                probes.append(dict(time=t, condition=label, velocity_rmse_per_band=errors.cpu().tolist(),
                                   endpoint_volume_mae=float((endpoint-target).abs().mean())))
        write_json(probe_file, dict(weight_identity=identity, case_id=row['case_id'],
                   donor_case_id=donor['case_id'], noise_seed=seed, times=list(TIMES), probes=probes))
        records.append(record)
        print(f"post-selection diagnostics {i+1}/10 | {row['case_id']}", flush=True)
        del source, wrong_source, target, layers, noise, features, z0, z1, state, residual, reference, endpoint
    # Regenerate the matched validation results from the SAME frozen best when needed.
    from report import evaluate_split
    selected_epoch = completion['best']['epoch']
    matched = evaluate_split(model, inputs, 'validation', folder / 'matched_oct', config, device, identity,
                            check_pause=lambda n: pause_at_boundary(root, 'diagnostics_matched', patients=n))
    wrong_mean = macro([r['metrics'] for r in records])
    delta = {g: {m: wrong_mean[g][m] - matched['mean'][g][m] for m in ('mae', 'psnr', 'ssim')} for g in GROUPS}
    summary = dict(phase='post_selection_validation_diagnostic', time=now(), patient_count=10,
                   patient_ids=[r['case_id'] for r in rows], weight_identity=identity,
                   best=completion['best'], solver=config['solver'], mean=wrong_mean,
                   matched_mean=matched['mean'], mismatched_minus_matched=delta, times=list(TIMES),
                   selection_use=False, retain_volumes=False,
                   patient_sd={g: {m: float(np.std([r['metrics'][g][m] for r in records], ddof=1))
                                   for m in ('mae', 'psnr', 'ssim')} for g in GROUPS})
    write_tables(folder / 'mismatched_oct', records, summary)
    write_json(folder / 'summary.json', summary)
    return summary

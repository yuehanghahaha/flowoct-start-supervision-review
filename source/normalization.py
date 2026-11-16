"""Resume-safe frozen training-only RMS statistics; no validation/test reads."""
import numpy as np
import torch
from dataset import Inputs
from fm3d.wavelet import haar, BANDS
from fm3d.flow import slab_max
from runtime import ROOT, now, read_json, write_json, sha256, pause_at_boundary


def prepare(config, binding):
    inputs = Inputs(binding, config, {'train'})
    rows = inputs.split('train')
    identity = dict(membership_sha256=sha256(ROOT / 'membership.json'),
                    deployment_sha256=sha256(ROOT / 'runs/deployment.json'),
                    case_ids=[r['case_id'] for r in rows], full_shape=config['full_shape'],
                    intensity_scale=config['intensity_scale'], scale_floor=config['loss']['scale_floor'])
    final = ROOT / 'runs/normalization.json'
    if final.is_file():
        stats = read_json(final)
        if stats['identity'] != identity or stats['patients'] != 140:
            raise ValueError('Frozen training statistics differ from data/protocol.')
        return stats
    partial = ROOT / 'runs/normalization.partial.json'
    progress = read_json(partial) if partial.is_file() else dict(identity=identity, completed=[],
                 band_sum_sq=[0.] * 8, band_count=0, sums=[0.] * 3, counts=[0] * 3, inputs={})
    if progress['identity'] != identity:
        raise ValueError('Partial statistics belong to another training set.')
    torch.set_num_threads(config['cpu_threads'])
    for index, row in enumerate(rows):
        if index < len(progress['completed']):
            if progress['completed'][index] != row['case_id']:
                raise ValueError('Statistics patient ordering changed.')
            continue
        pause_at_boundary(ROOT, 'statistics', patients=index, update=0)
        target = inputs.volume(row, 'octa')
        layers = torch.from_numpy(inputs.layers(row))[None]
        bands = haar(target).numpy()[0]
        sumsq = np.einsum('bzyx,bzyx->b', bands, bands, dtype=np.float64, optimize=False)
        progress['band_sum_sq'] = (np.array(progress['band_sum_sq']) + sumsq).tolist()
        progress['band_count'] += int(bands[0].size)
        arrays = [target.numpy()]
        for lower, upper in ((0, 2), (2, 5)):
            projection, valid = slab_max(target, layers, lower, upper, config['native_layer_depth'])
            arrays.append(projection[valid].numpy())
        for k, a in enumerate(arrays):
            flat = a.ravel()
            progress['sums'][k] += float(np.einsum('i,i->', flat, flat, dtype=np.float64, optimize=False))
            progress['counts'][k] += int(flat.size)
        progress['completed'].append(row['case_id'])
        progress['inputs'][row['case_id']] = {k: inputs.identify(row, k) for k in ('image', 'layers')}
        write_json(partial, progress)
        print(f"training-only RMS {index + 1}/140 | {row['case_id']}", flush=True)
        del bands, arrays, target, layers
    floor = config['loss']['scale_floor']
    if len(progress['completed']) != 140 or min(progress['counts']) < 1:
        raise ValueError('Incomplete training-only statistics.')
    band = np.maximum(np.sqrt(np.array(progress['band_sum_sq']) / progress['band_count']), floor)
    scale = np.maximum(np.sqrt(np.array(progress['sums']) / np.array(progress['counts'])), floor)
    stats = dict(identity=identity, patients=140, band_order=list(BANDS), band_rms=band.tolist(),
                 volume_rms=float(scale[0]), ilm_opl_max_rms=float(scale[1]), opl_bm_max_rms=float(scale[2]),
                 accumulation_dtype='float64', transform_dtype='float32', time=now(),
                 band_count=progress['band_count'], counts=progress['counts'], inputs=progress['inputs'])
    write_json(final, stats)
    # Partial numerical evidence is retained; no intermediate 3D arrays are saved.
    return stats

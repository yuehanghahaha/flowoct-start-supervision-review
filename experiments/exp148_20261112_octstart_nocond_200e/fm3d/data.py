import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def read_rows(path):
    path = Path(path).resolve()
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty CSV')
    for row in rows:
        for key in ('case_id', 'patient_id', 'path'):
            if not row.get(key, '').strip():
                raise ValueError(f'Missing {key}; patient_id must come from verified metadata')
        row['path'] = str((path.parent / row['path']).resolve())
    return rows


def load_pair(path, axis_order, intensity_scale, depth):
    """Resize in stored intensity units, then fixed-scale before paired cropping."""
    if sorted(axis_order) != ['X', 'Y', 'Z']:
        raise ValueError('axis_order must be a permutation of ZYX')
    if not math.isfinite(intensity_scale) or intensity_scale <= 0 or depth < 0:
        raise ValueError('intensity_scale must be finite and >0; depth must be >=0')
    tensors = []
    with np.load(path, allow_pickle=False) as data:
        if 'axis_order' in data and str(data['axis_order'].item()) != axis_order:
            raise ValueError(f'NPZ axis metadata conflicts with configuration: {path}')
        for key in ('oct', 'octa'):
            a = np.asarray(data[key], dtype=np.float32)
            if a.ndim != 3 or min(a.shape) < 8 or not np.isfinite(a).all():
                raise ValueError(f'{path}: {key} must be finite 3D with dimensions >=8')
            if a.min() < 0 or a.max() > intensity_scale:
                raise ValueError(f'{path}: {key} outside [0,{intensity_scale}]; verify scale')
            a = np.transpose(a, [axis_order.index(axis) for axis in 'ZYX']).copy()
            tensors.append(torch.from_numpy(a)[None])
    source, target = tensors
    if source.shape != target.shape:
        raise ValueError(f'Unaligned OCT/OCTA shapes: {path}')
    if depth and source.shape[1] != depth:
        size = (depth, *source.shape[2:])
        source, target = [F.interpolate(x[None], size=size, mode='trilinear',
                                       align_corners=False)[0] for x in tensors]
    return (source / intensity_scale).contiguous(), (target / intensity_scale).contiguous()

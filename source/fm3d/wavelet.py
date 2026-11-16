"""One orthonormal 3D Haar level, in ZYX subband order."""
import math

import torch


BANDS = ("LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH")
SCALE = 1 / math.sqrt(2)


def split_axis(x, axis):
    even = [slice(None)] * x.ndim
    odd = even.copy()
    even[axis] = slice(None, None, 2)
    odd[axis] = slice(1, None, 2)
    a, b = x[tuple(even)], x[tuple(odd)]
    return (a + b) * SCALE, (b - a) * SCALE


def haar(volume):
    if volume.ndim != 5 or volume.shape[1] != 1 or any(n % 2 for n in volume.shape[2:]):
        raise ValueError("Haar expects [B,1,Z,Y,X] with even spatial sizes.")
    bands = [volume]
    for axis in (2, 3, 4):
        bands = [part for band in bands for part in split_axis(band, axis)]
    return torch.cat(bands, dim=1)


def inverse_haar(coefficients):
    if coefficients.ndim != 5 or coefficients.shape[1] != 8:
        raise ValueError("Inverse Haar expects eight ZYX-ordered subbands.")
    bands = coefficients.split(1, dim=1)
    for axis in (4, 3, 2):
        bands = [
            torch.stack(((low - high) * SCALE, (low + high) * SCALE), dim=axis + 1).flatten(axis, axis + 1)
            for low, high in zip(bands[::2], bands[1::2])
        ]
    return bands[0]

import numpy as np


def project(volume, layers):
    """Layer is [6,Y,X], MATLAB 1-based native 640-depth boundary coordinates.

    Select resampled voxel centers inside closed layer intervals. Use the same
    supplied annotation for prediction and reference, for post-hoc evaluation only.
    Returned projection orientation is [X,Y], matching the distributed BMPs.
    """
    if volume.ndim != 3 or layers.shape != (6, *volume.shape[1:]):
        raise ValueError('Expected volume ZYX and layers 6YX')
    if not np.isfinite(volume).all() or not np.isfinite(layers).all():
        raise ValueError('Nonfinite input')
    if (layers < 1).any() or (layers > 640).any() or (np.diff(layers.astype(float), axis=0) < 0).any():
        raise ValueError('Invalid native layer boundaries')
    centers = ((np.arange(volume.shape[0]) + 0.5) * 640 / volume.shape[0] - 0.5)[:, None, None]
    maps, empty = {'FULL': volume.mean(0, dtype=np.float64).T.astype(np.float32)}, {}
    for name, lower, upper in (('ILM_OPL', 0, 2), ('OPL_BM', 2, 5)):
        mask = (centers >= layers[lower].astype(float) - 1) & (
            centers <= layers[upper].astype(float) - 1)
        count = mask.sum(0)
        empty[name] = int((count == 0).sum())
        # Zero for a ray with no samples in the interval; disclose its count.
        maps[name] = np.where(mask, volume, 0).max(0).T
    return maps, empty

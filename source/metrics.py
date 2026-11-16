"""Complete 3D, slice-wise 2D, and separately named projection metrics."""
import math

import numpy as np
from skimage.metrics import structural_similarity

from fm3d.projection import project


GROUPS = ("volume", "bscan2d", "ilm_opl_max", "opl_bm_max", "full_mean", "full_max",
          "ilm_opl_mean", "opl_bm_mean", "ilm_bm_mean")
METRICS = ("mae", "psnr", "ssim")
PROTOCOL = {
    "normalized_intensity": "canonical Z256 float32 values divided by 255; no image-wise stretching",
    "full_volume": "all 256x304x304 voxels, 3D SSIM",
    "bscan2d": "all Y-axis ZX slices; each metric computed per slice then averaged within patient",
    "main_projection": "ILM-OPL maximum; native640 MATLAB boundaries and closed voxel-center intervals",
    "other_projections": "OPL-BM max; full-Z mean/max; ILM-OPL, OPL-BM, ILM-BM means",
    "slab_means": "native640 centers, ILM<=z<OPL, OPL<=z<=BM, ILM<=z<=BM; count+1e-6",
    "projection_orientation": "XY; projections computed after generation using identical GT layers",
    "range": 1.0, "ssim_window": 7, "ssim_weights": "uniform", "ssim_sample_covariance": True,
    "psnr": "-10*log10(max(MSE,1e-12)); patient macro average, not pooled-voxel PSNR",
    "aggregation": "equal patient macro average; no test-based checkpoint or solver selection",
}


def scores(pred, truth):
    abs_sum = sq_sum = 0.0
    for offset in range(0, truth.shape[0], 16):
        error = pred[offset:offset + 16].astype(np.float64) - truth[offset:offset + 16]
        abs_sum += float(np.abs(error).sum())
        sq_sum += float(np.square(error).sum())
    return dict(mae=abs_sum / truth.size, psnr=-10 * math.log10(max(sq_sum / truth.size, 1e-12)),
                ssim=float(structural_similarity(truth, pred, data_range=1., win_size=7,
                      gaussian_weights=False, use_sample_covariance=True, channel_axis=None)))


def projection_maps(volume, layers):
    views, empty = project(volume, layers)
    maps = dict(ilm_opl_max=views["ILM_OPL"], opl_bm_max=views["OPL_BM"],
                full_mean=views["FULL"], full_max=volume.max(0).T)
    empty = dict(ilm_opl_max=empty["ILM_OPL"], opl_bm_max=empty["OPL_BM"])
    centers = ((np.arange(volume.shape[0], dtype=np.float32) + .5) * 640 / volume.shape[0] - .5)[:, None, None]
    ilm, opl, bm = [layers[i] - 1 for i in (0, 2, 5)]
    for name, low, high, open_high in (("ilm_opl_mean", ilm, opl, True),
                                      ("opl_bm_mean", opl, bm, False),
                                      ("ilm_bm_mean", ilm, bm, False)):
        mask = (centers >= low) & ((centers < high) if open_high else (centers <= high))
        count = mask.sum(0)
        maps[name] = (np.where(mask, volume, 0).sum(0, dtype=np.float32) / (count + 1e-6)).T.astype(np.float32)
        empty[name] = int((count == 0).sum())
    return maps, empty


def evaluate(pred, truth, layers):
    if pred.shape != truth.shape or pred.ndim != 3 or min(pred.shape) < 7:
        raise ValueError("Aligned full-resolution 3D volumes are required.")
    values = {"volume": scores(pred, truth)}
    slices = [dict(y=y, **scores(pred[:, y, :], truth[:, y, :])) for y in range(truth.shape[1])]
    values["bscan2d"] = {key: float(np.mean([row[key] for row in slices])) for key in METRICS}
    p, empty = projection_maps(pred, layers)
    g, _ = projection_maps(truth, layers)
    values.update({name: scores(p[name], g[name]) for name in GROUPS[2:]})
    return values, slices, p, g, empty


def macro(values):
    if not values:
        raise ValueError("An empty patient split cannot be aggregated.")
    return {group: {metric: float(np.mean([item[group][metric] for item in values]))
                    for metric in METRICS} for group in GROUPS}

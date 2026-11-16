"""Eight paper-named metrics under the explicitly documented operational convention."""
import math

import numpy as np
from skimage.metrics import structural_similarity

from dataset import unit

NAMES = ("MAE", "PSNR", "SSIM", "MAE-V", "PSNR-V", "SSIM-V", "VDE", "VDC")


def image_metrics(prediction, target, config):
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    difference = prediction - target
    mse = float(np.mean(difference * difference))
    options = config["ssim"]
    return {"MAE": float(np.mean(np.abs(difference))),
            "PSNR": float("inf") if mse == 0 else 10 * math.log10(config["metric_data_range"] ** 2 / mse),
            "SSIM": 100 * float(structural_similarity(target, prediction, data_range=config["metric_data_range"], **options))}


def density_metrics(prediction_projection, target_projection, patch=16):
    pred = np.asarray(prediction_projection) > np.mean(prediction_projection)
    true = np.asarray(target_projection) > np.mean(target_projection)
    if pred.shape != true.shape or any(n % patch for n in pred.shape):
        raise ValueError("Projection dimensions must match and divide into complete 16x16 blocks")
    def densities(mask):
        h, w = mask.shape
        return mask.reshape(h // patch, patch, w // patch, patch).mean(axis=(1, 3)).ravel()
    a, b = densities(pred), densities(true)
    correlation = None if np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1] * 100)
    return {"VDE": float(abs(pred.mean() - true.mean())), "VDC": correlation}, {"pred_mask": pred, "gt_mask": true}


def eight_metrics(prediction_zyx, target_zyx, vessel_mask_yx, config):
    if prediction_zyx.shape != target_zyx.shape or prediction_zyx.ndim != 3:
        raise ValueError("Expected aligned full ZYX volumes")
    if not np.isfinite(prediction_zyx).all() or not np.isfinite(target_zyx).all():
        raise ValueError("Non-finite prediction/target")
    if prediction_zyx.min() < 0 or prediction_zyx.max() > 1 or target_zyx.min() < 0 or target_zyx.max() > 1:
        raise ValueError("Evaluation convention is float unit range")
    slices = [image_metrics(prediction_zyx[:, y, :], target_zyx[:, y, :], config) for y in range(target_zyx.shape[1])]
    result = {name: float(np.mean([row[name] for row in slices])) for name in ("MAE", "PSNR", "SSIM")}
    pred_projection = unit(prediction_zyx.mean(axis=0))
    gt_projection = unit(target_zyx.mean(axis=0))
    if vessel_mask_yx.shape != gt_projection.shape or vessel_mask_yx.dtype != np.bool_:
        raise ValueError("Expected a registered boolean GT mask in YX orientation")
    weights = np.where(vessel_mask_yx, 1.0, config["background_gamma"])
    weighted = image_metrics(pred_projection * weights, gt_projection * weights, config)
    result.update({name + "-V": value for name, value in weighted.items()})
    density, masks = density_metrics(pred_projection, gt_projection, config["vdc_patch"])
    result.update(density)
    views = {"prediction_yx": pred_projection, "target_yx": gt_projection,
             "absolute_error_yx": np.abs(pred_projection - gt_projection),
             "annotation_yx": vessel_mask_yx, **masks}
    return result, slices, views


def aggregate(rows):
    means, valid = {}, {}
    for name in NAMES:
        values = [float(row[name]) for row in rows if row[name] is not None]
        means[name] = float(np.mean(values)) if values else None
        valid[name] = len(values)
    return {"patient_mean": means, "valid_patients": valid, "patients": len(rows),
            "undefined_note": "Constant density arrays produce undefined VDC; these cases and counts are reported, not replaced with zero."}

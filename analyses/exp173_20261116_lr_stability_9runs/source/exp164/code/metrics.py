"""Separate 3D, B-scan, MIP, and XOCT-style evaluation objects.

Main metrics use unquantized [0,1] data and a fixed range of 1. All SSIMs
use skimage win7, uniform weights, sample covariance, channel_axis=None.
XOCT-style supplemental metrics reproduce the released metric arithmetic,
but use this project's grid and documented layer rasterization.
"""
import math

import numpy as np
from skimage.metrics import structural_similarity

from fm3d.projection_report import project


PROTOCOL = dict(
    normalization='parent fixed /255 after Z256 trilinear align_corners=False; no image stretching',
    volume='full ZYX256x304x304, float32, range1; MAE/PSNR/3D SSIM',
    bscan='each Y slice [Z,X], all304 slices; MAE/PSNR/2D SSIM per slice then volume then patient',
    bscan_unpublished_choices='fixed range1, float32, win7 uniform sample covariance; no uint8; not exact TransPro table reproduction',
    main_projection='ILM-OPL maximum; native640 voxel centers and closed intervals, inherited exp018',
    supplementary_projection='OPL-BM maximum, full-Z mean, full-Z maximum; MAE/PSNR/2D SSIM each',
    xoct_float='ILM-BM/ILM-OPL/OPL-BM masked means and full-Z mean; unquantized range1',
    xoctstyle='floor(255*x) float32 operands, half-open ILM<=z<OPL and OPL<=z<=BM; masked mean denominator count+1e-6; GT-range normalize each object to255 and clip prediction; MAE units0..255, PSNR/SSIM use GT max-min',
    xoctstyle_limit='XOCT eval3d.py/util/eval.py arithmetic on project Z256 grid and fixed split; original ready-made segmentation conversion unpublished; no claim of identical paper protocol; Perp not included',
    ssim='3D for volume, 2D for every projection and B-scan; win7 uniform, use_sample_covariance=True, channel_axis=None',
    psnr='-10log10(max(MSE,1e-12))+20log10(range); zero-error cap disclosed (XOCT reference gives +inf)',
    empty_slab='zero at empty ray for projection evaluation; count reported; training excludes empty rays',
    reference_volume='additional unquantized FlowOCT GT.max-GT.min PSNR and 3D SSIM',
    aggregation='volume -> within patient -> equal patient macro; all slices before patient macro; no pooled-voxel PSNR',
    regions='diagnostic ILM-OPL MIP errors; not TransPro vessel-weighted MAE-V/PSNR-V/SSIM-V')


def scores(pred, gt, data_range=1., with_pcc=False):
    abs_sum = sq_sum = 0.
    for first in range(0, gt.shape[0], 16):
        error = pred[first:first+16].astype(np.float64) - gt[first:first+16].astype(np.float64)
        abs_sum += float(np.abs(error).sum())
        sq_sum += float(np.square(error).sum())
    mse = sq_sum / gt.size
    result = dict(mae=abs_sum/gt.size, mse=mse,
                  psnr=-10*math.log10(max(mse, 1e-12)) + 20*math.log10(data_range),
                  ssim=float(structural_similarity(gt, pred, data_range=data_range, win_size=7,
                             gaussian_weights=False, use_sample_covariance=True, channel_axis=None)))
    if with_pcc:
        a = pred.astype(np.float64) - pred.mean(dtype=np.float64)
        b = gt.astype(np.float64) - gt.mean(dtype=np.float64)
        denominator = np.sqrt(np.square(a).sum() * np.square(b).sum())
        result['pcc'] = float((a*b).sum()/denominator) if denominator > 1e-15 else None
    return result


def slab_means(volume, layers):
    """Author-style label rasterization, explicit unpublished boundary choice."""
    centers = ((np.arange(volume.shape[0], dtype=np.float32)+.5)*640/volume.shape[0]-.5)[:, None, None]
    ilm, opl, bm = [layers[index]-1 for index in (0, 2, 5)]
    masks = dict(ILM_OPL_MEAN=(centers >= ilm) & (centers < opl),
                 OPL_BM_MEAN=(centers >= opl) & (centers <= bm),
                 ILM_BM_MEAN=(centers >= ilm) & (centers <= bm))
    views, empty = {}, {}
    for name, mask in masks.items():
        count = mask.sum(0)
        views[name] = (np.where(mask, volume, 0).sum(0, dtype=np.float32)/(count+1e-6)).T.astype(np.float32)
        empty[name] = int((count == 0).sum())
    return views, empty


def xoctstyle_scores(pred, gt):
    extent = float(gt.max()-gt.min())
    if extent <= 0:
        return dict(mae=None, mse=None, psnr=None, ssim=None)
    # This mirrors the author normalization; float32 avoids unsigned overflow.
    low = gt.min()
    truth = (gt-low)/extent * 255
    estimate = ((pred-low)/extent).clip(0, 1) * 255
    return scores(estimate, truth, float(truth.max()-truth.min()))


def bscan_scores(pred, gt):
    rows = [dict(y=index, **scores(pred[:, index, :], gt[:, index, :]))
            for index in range(gt.shape[1])]
    mean = {key: float(np.mean([row[key] for row in rows])) for key in ('mae', 'mse', 'psnr', 'ssim')}
    return mean, rows


def quality(pred, gt, layers, regions, include_reference=True):
    if pred.shape != gt.shape or pred.ndim != 3 or min(pred.shape) < 7:
        raise ValueError('Aligned 3D volumes of dimensions>=7 required')
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise FloatingPointError('Nonfinite evaluation input')
    metrics = {'volume_'+key: value for key, value in scores(pred, gt).items()}
    data_range = float(gt.max()-gt.min())
    metrics['reference_volume_data_range'] = data_range
    if data_range > 0:
        metrics['reference_volume_psnr'] = metrics['volume_psnr'] + 20*math.log10(data_range)
        metrics['reference_volume_ssim'] = metrics['volume_ssim'] if data_range == 1 else float(
            structural_similarity(gt, pred, data_range=data_range, win_size=7))
    else:
        metrics.update(reference_volume_psnr=None, reference_volume_ssim=None)
    bscan, slices = bscan_scores(pred, gt)
    metrics.update({'bscan2d_'+key: value for key, value in bscan.items()})
    p, empty = project(pred, layers)
    g, _ = project(gt, layers)
    p['ZMAX'], g['ZMAX'] = pred.max(0).T, gt.max(0).T
    extra, mean_empty = slab_means(pred, layers)
    p.update(extra)
    g.update(slab_means(gt, layers)[0])
    for name in p:
        metrics.update({'projection_'+name.lower()+'_'+key: value
                        for key, value in scores(p[name], g[name], with_pcc=True).items()})
    metrics['ilm_opl_empty_rays'] = empty['ILM_OPL']
    metrics['opl_bm_empty_rays'] = empty['OPL_BM']
    metrics.update({name.lower()+'_empty_rays': count for name, count in mean_empty.items()})
    error = np.abs(p['ILM_OPL'].astype(np.float64)-g['ILM_OPL'])
    for name, mask in regions.items():
        metrics['region_'+name+'_pixels'] = int(mask.sum())
        metrics['region_'+name+'_mae'] = float(error[mask].mean()) if mask.any() else None
        metrics['region_'+name+'_bias'] = float((p['ILM_OPL']-g['ILM_OPL'])[mask].mean(dtype=np.float64)) if mask.any() else None
    # Supplemental XOCT metric group is evaluated only after a checkpoint is
    # frozen for formal testing / offline reevaluation; never used for selection.
    if include_reference:
        uq_pred = (pred.clip(0, 1)*255).astype(np.uint8).astype(np.float32)
        uq_gt = (gt.clip(0, 1)*255).astype(np.uint8).astype(np.float32)
        qp = slab_means(uq_pred, layers)[0]
        qg = slab_means(uq_gt, layers)[0]
        qp['FULL_MEAN'], qg['FULL_MEAN'] = uq_pred.mean(0).T, uq_gt.mean(0).T
        qp['VOLUME'], qg['VOLUME'] = uq_pred, uq_gt
        for name in qp:
            metrics.update({'xoctstyle_'+name.lower()+'_'+key: value
                            for key, value in xoctstyle_scores(qp[name], qg[name]).items()})
    return metrics, p, g, slices

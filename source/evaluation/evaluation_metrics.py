"""Numerical functions inherited from exp003, unchanged scientific definitions."""
import math
import numpy as np
from fm3d.projection_report import project
from fm3d.train_start import patient_mean,map_scores


def summarize(records, settings):
    report = patient_mean(records)
    rng = np.random.default_rng(settings['bootstrap_seed'])
    patients = list(report['per_patient'].values())
    draw = rng.integers(len(patients), size=(settings['bootstrap_repetitions'], len(patients)))
    stats = {}
    for key in report['metrics']:
        values = np.array([p[key] for p in patients if p[key] is not None], dtype=float)
        if not len(values):
            stats[key] = None
            continue
        ci = np.percentile(values[draw].mean(1), [2.5, 97.5]) if len(values) == len(patients) else [None, None]
        stats[key] = dict(mean=float(values.mean()), sample_sd=float(values.std(ddof=1)) if len(values)>1 else 0.,
            median=float(np.median(values)), min=float(values.min()), max=float(values.max()),
            bootstrap95=list(map(float, ci)) if ci[0] is not None else [None, None])
    report['distribution'] = stats
    return report


def additional_metrics(prediction, target, layers, threshold):
    p, g = prediction[0, 0].numpy(), target[0, 0].numpy()
    pa, pe = project(p, layers)
    ga, ge = project(g, layers)
    result = {}
    for label, key in (('full_mean', 'FULL'), ('opl_bm', 'OPL_BM')):
        scores = map_scores(pa[key], ga[key])
        mse = float(np.mean((pa[key].astype(np.float64)-ga[key])**2))
        scores['psnr'] = -10*math.log10(max(mse, 1e-12))
        result.update({label+'_'+k: v for k, v in scores.items()})
    high = g > threshold
    error = np.abs(p-g)
    result.update(high_intensity_fraction=float(high.mean()), pred_high_intensity_fraction=float((p>threshold).mean()),
        high_intensity_mae=float(error[high].mean(dtype=np.float64)) if high.any() else None,
        background_mae=float(error[~high].mean(dtype=np.float64)) if (~high).any() else None,
        high_intensity_gt_mean=float(g[high].mean(dtype=np.float64)) if high.any() else None,
        high_intensity_pred_mean=float(p[high].mean(dtype=np.float64)) if high.any() else None,
        ilm_opl_pred_std=float(pa['ILM_OPL'].std(dtype=np.float64)),
        ilm_opl_gt_std=float(ga['ILM_OPL'].std(dtype=np.float64)),
        ilm_opl_pred_mean=float(pa['ILM_OPL'].mean(dtype=np.float64)),
        ilm_opl_gt_mean=float(ga['ILM_OPL'].mean(dtype=np.float64)),
        opl_bm_empty_rays=pe['OPL_BM'])
    return result, pa, ga


def save_views(path, prediction, source, target, pa, ga):
    p, s, g = [v[0, 0].numpy() for v in (prediction, source, target)]
    arrays = dict(volume_shape_ZYX=np.array(g.shape), source_full_mean=s.mean(0).T,
                  pred_zmax=p.max(0).T, gt_zmax=g.max(0).T)
    for prefix, maps in (('pred', pa), ('gt', ga)):
        for label, key in (('full_mean','FULL'), ('ilm_opl','ILM_OPL'), ('opl_bm','OPL_BM')):
            arrays[prefix+'_'+label] = maps[key]
    positions = [min(int(g.shape[1]*f), g.shape[1]-1) for f in (.25, .5, .75)]
    arrays['bscan_y_positions'] = np.array(positions)
    for i, y in enumerate(positions):
        for prefix, value in (('pred',p),('gt',g),('source',s)):
            arrays[f'bscan_{prefix}_{i}'] = value[:,y,:]
    np.savez_compressed(path, **arrays)

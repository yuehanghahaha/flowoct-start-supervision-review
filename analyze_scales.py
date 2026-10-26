"""Descriptive intensity/scale analysis of all 50 saved test projections.

These statistics are not vessel segmentation, diameter, recall, or causality tests.
Gaussian filtering is used for diagnostics only; the published predictions stay raw.
"""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

ROOT = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return float(np.mean(values)) if values else None


def main():
    config = json.loads((ROOT / 'experiment.json').read_text(encoding='utf-8'))
    manifest = json.loads((ROOT / 'verification/derived_data.json').read_text(encoding='utf-8'))
    out = ROOT / 'report'
    figdir = out / 'figures'
    figdir.mkdir(exist_ok=True)
    bins, scales = [], []
    for case in manifest['cases']:
        path = ROOT / case['derived_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == case['derived_sha256']
        with np.load(path, allow_pickle=False) as arrays:
            gt, pred = [arrays[key].astype(np.float64) for key in ['gt', 'pred']]
        assert gt.shape == pred.shape == (304, 304)
        assert np.isfinite(gt).all() and np.isfinite(pred).all()
        edges = config['intensity_bins']
        for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (gt >= lower) & ((gt <= upper) if i == len(edges) - 2 else (gt < upper))
            count = int(mask.sum())
            bins.append(dict(case_id=case['case_id'], bin_index=i, lower=lower, upper=upper,
                             pixels=count, gt_mean=float(gt[mask].mean()) if count else None,
                             pred_mean=float(pred[mask].mean()) if count else None,
                             bias=float((pred - gt)[mask].mean()) if count else None,
                             mae=float(np.abs(pred - gt)[mask].mean()) if count else None))
        for sigma in config['gaussian_sigmas_pixels']:
            g_low, p_low = [gaussian_filter(x, sigma=sigma, mode='reflect') for x in [gt, pred]]
            g_high, p_high = gt - g_low, pred - p_low
            scales.append(dict(case_id=case['case_id'], sigma_pixels=sigma,
                gt_low_std=float(g_low.std()), pred_low_std=float(p_low.std()),
                gt_high_std=float(g_high.std()), pred_high_std=float(p_high.std()),
                low_std_ratio=float(p_low.std() / g_low.std()),
                high_std_ratio=float(p_high.std() / g_high.std()),
                high_mae=float(np.abs(p_high - g_high).mean()),
                gt_display_saturated_fraction=float((gt > .5).mean()),
                pred_display_saturated_fraction=float((pred > .5).mean())))
    summary_bins = []
    for i in range(len(config['intensity_bins']) - 1):
        rows = [row for row in bins if row['bin_index'] == i and row['pixels']]
        summary_bins.append(dict(bin_index=i, lower=config['intensity_bins'][i],
            upper=config['intensity_bins'][i + 1], valid_patients=len(rows),
            gt_mean=mean(rows, 'gt_mean'), pred_mean=mean(rows, 'pred_mean'),
            bias=mean(rows, 'bias'), mae=mean(rows, 'mae'),
            pred_mean_patient_sd=float(np.std([r['pred_mean'] for r in rows], ddof=1))
                if len(rows) > 1 else None))
    summary_scales = []
    for sigma in config['gaussian_sigmas_pixels']:
        rows = [row for row in scales if row['sigma_pixels'] == sigma]
        summary_scales.append(dict(sigma_pixels=sigma, patients=len(rows),
            low_std_ratio=mean(rows, 'low_std_ratio'), high_std_ratio=mean(rows, 'high_std_ratio'),
            high_mae=mean(rows, 'high_mae'),
            patients_high_std_ratio_below_one=sum(row['high_std_ratio'] < 1 for row in rows)))
    write_csv(out / 'intensity_bins_per_case.csv', bins)
    write_csv(out / 'scale_per_case.csv', scales)
    summary = dict(patients=50, calibration=summary_bins, scales=summary_scales,
        interpretation='Post-hoc descriptive diagnostics only; intensity is not caliber, high-frequency signal includes noise.',
        segmentation_labels_used=False, training_or_model_selection=False)
    (out / 'scale_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    x = [r['gt_mean'] for r in summary_bins]
    y = [r['pred_mean'] for r in summary_bins]
    axes[0].errorbar(x, y, yerr=[r['pred_mean_patient_sd'] for r in summary_bins],
                     fmt='o-', capsize=3, color='#177984', label='Patient mean +/- SD')
    axes[0].plot([0, 1], [0, 1], '--', color='#8b959c')
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='GT intensity-bin mean',
                ylabel='Prediction at the same GT-defined pixels', title='Intensity calibration, all 50 cases')
    axes[0].legend(fontsize=8)
    pos = np.arange(len(summary_scales))
    axes[1].bar(pos - .18, [r['low_std_ratio'] for r in summary_scales], width=.36,
                 color='#177984', label='Low-pass std ratio')
    axes[1].bar(pos + .18, [r['high_std_ratio'] for r in summary_scales], width=.36,
                 color='#bd813d', label='High-pass std ratio')
    axes[1].axhline(1, color='#8b959c', linestyle='--')
    axes[1].set(xticks=pos, xticklabels=config['gaussian_sigmas_pixels'], xlabel='Gaussian sigma / pixels',
                ylabel='Prediction / GT, equal patient mean', title='Spatial signal, not vessel recall')
    axes[1].legend(fontsize=8)
    rows = [r for r in scales if r['sigma_pixels'] == 1]
    axes[2].scatter([r['low_std_ratio'] for r in rows], [r['high_std_ratio'] for r in rows],
                    s=22, color='#177984', alpha=.8)
    axes[2].axhline(1, color='#8b959c', linestyle='--')
    axes[2].axvline(1, color='#8b959c', linestyle='--')
    axes[2].set(xlabel='Low-pass std ratio', ylabel='High-pass std ratio',
                title='Each point is one test patient; sigma=1')
    fig.savefig(figdir / 'scale_diagnostics.png', dpi=170, bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), constrained_layout=True)
    for row, cid in enumerate(config['fixed_visual_examples']):
        with np.load(ROOT / 'derived/ilm_opl' / f'{cid}.npz', allow_pickle=False) as arrays:
            gt, pred = arrays['gt'], arrays['pred']
        for col, (array, limit, title) in enumerate([
            (gt, 1, 'GT full [0,1]'), (pred, 1, 'Prediction full [0,1]'),
            (gt[88:216, 88:216], .5, 'GT zoom [0,0.5]'),
            (pred[88:216, 88:216], .5, 'Prediction zoom [0,0.5]')]):
            axes[row, col].imshow(array, cmap='gray', vmin=0, vmax=limit, interpolation='nearest')
            axes[row, col].set_title(f'{cid}: {title}', fontsize=9)
            axes[row, col].axis('off')
    fig.suptitle('Raw shallow projections; shared display limits, no sharpening or per-image stretching')
    fig.savefig(figdir / 'fixed_examples.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()

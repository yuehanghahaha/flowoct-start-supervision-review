"""Build reproducible figures, a result summary and complete tables from committed evidence."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'analysis'
METRICS = ('mae', 'psnr', 'ssim')
OBJECTS = ('volume', 'bscan2d', 'ILM_OPL', 'OPL_BM', 'FULL', 'ZMAX',
           'ILM_OPL_MEAN', 'OPL_BM_MEAN', 'ILM_BM_MEAN')
COLORS = {'E090': '#cb423c', 'E120': '#316ea4', 'E130': '#2d8b63',
          'E190': '#8e5abb', 'XOCT_E213': '#d78738', 'GT': '#20252a'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def collect(pattern):
    return [load(path) for path in sorted(ROOT.glob(pattern))]


def table(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def means(rows, keys, values):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    result = []
    for group, subset in groups.items():
        counts = len({r['case'] for r in subset})
        assert counts == len(subset), group
        result.append(dict(zip(keys, group), patients=counts, **{
            name: float(np.mean(valid)) if (valid := [r[name] for r in subset if r.get(name) is not None]) else None
            for name in values}))
    return result


def score_rows(rows, data, arm, treatment):
    for obj, scores in data['standard'].items():
        rows.append(dict(model=arm['model'], case=arm['case'], treatment=treatment, object=obj,
                         **{name: scores[name] for name in METRICS}))


def oracle_summary(core):
    rows = []
    for r in core:
        baseline = r['endpoint']['standard']['volume']
        rows += [dict(model=r['model'], case=r['case'], correction=name,
                     **{f'delta_{key}': result['standard']['volume'][key]-baseline[key]
                        for key in METRICS}) for name, result in r['oracle'].items()]
    return means(rows, ('model', 'correction'), tuple(f'delta_{name}' for name in METRICS))


def find_cases(rows):
    by_model = defaultdict(list)
    for row in rows:
        by_model[row['model']].append(row)
    if not by_model:
        return {}
    matched = set.intersection(*({r['case'] for r in rows} for rows in by_model.values()))
    return {name: [r for r in rows if r['case'] in matched] for name, rows in by_model.items()}


def render(fig, name):
    path = OUT/'figures'/name
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def curves(core):
    groups = find_cases(core)
    if not groups or not any(groups.values()):
        return
    patients = next(len(rows) for rows in groups.values() if rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    sample = next(r for rows in groups.values() for r in rows)
    bins = np.asarray(sample['histogram_edges'])
    centers = (bins[1:] + bins[:-1])/2
    gt_records = [load(ROOT/'reports/truth'/r['case']/'record.json') for r in next(iter(groups.values()))]
    counts = np.mean([r['distribution']['histogram'] for r in gt_records], axis=0)
    axes[0, 0].plot(centers, np.maximum(counts/counts.sum(), 1e-8), 'k-', label='GT')
    gt_curve = np.mean([r['clipped']['by_depth']['gt_mean'] for r in next(iter(groups.values()))], axis=0)
    axes[0, 1].plot(gt_curve, 'k-', label='GT')
    region_labels = ['retina_exterior', 'background_ILM_OPL', 'small_vessel_candidate_3d',
                     'large_vessel_candidate_3d', 'FAZ_ILM_OPL', 'FAZ3D_retinal']
    for j, (tag, rows) in enumerate(groups.items()):
        color = COLORS[tag]
        counts = np.mean([r['raw']['distribution']['histogram'] for r in rows], axis=0)
        axes[0, 0].plot(centers, np.maximum(counts/counts.sum(), 1e-8), color=color, label=tag)
        axes[0, 1].plot(np.mean([r['clipped']['by_depth']['mean'] for r in rows], axis=0), color=color, label=tag)
        axes[1, 0].plot(np.mean([r['clipped']['by_depth']['bias'] for r in rows], axis=0), color=color)
        values = []
        for region in region_labels:
            valid = [r['clipped']['regions'].get(region, {}).get('bias') for r in rows]
            sample = [v for v in valid if v is not None]
            values.append(np.mean(sample) if sample else np.nan)
        axes[1, 1].plot(np.arange(len(values)), values, marker='o', color=color, label=tag)
    axes[0, 0].set(yscale='log', title='A | Raw intensity histogram', xlim=(-.10, 1.05), ylabel='Fraction per bin')
    axes[0, 1].set(title='B | Intensity by axial plane', xlabel='Z256 plane', ylabel='Mean intensity')
    axes[1, 0].set(title='B | Signed bias by axial plane', xlabel='Z256 plane', ylabel='Prediction − GT')
    axes[1, 1].set(title='A | Signed bias in fixed GT regions', ylabel='Prediction − GT')
    axes[1, 1].set_xticks(np.arange(6), ['Outside\nretina', 'Shallow\nbackground', 'Small vessel\ncandidate',
                                       'Large vessel\ncandidate', '2D FAZ\nshallow', '3D FAZ\nretinal'])
    for ax in axes.flat:
        ax.grid(alpha=.18)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f'FM stability | Paired validation patients: {patients}')
    render(fig, 'intensity_depth.png')

    groups = find_cases([r for r in core if r['trajectory']])
    if not groups or not any(groups.values()):
        return
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for model, rows in groups.items():
        t = [r['t'] for r in rows[0]['trajectory']]
        cl = COLORS[model]
        for ax, key in zip(axs[0], ['mae', 'bias']):
            ax.plot(t, np.mean([[r[key] for r in c['trajectory']] for c in rows], axis=0),
                    'o-', color=cl, label=model, markersize=4)
        t = t[:-1]
        for flavor, ls in [('teacher_mse', '--'), ('free_mse', '-')]:
            axs[1, 0].plot(t, np.mean([[r[flavor] for r in c['velocity']] for c in rows], axis=0),
                           ls, color=cl, label=f'{model} {flavor.split("_")[0]}')
        axs[1, 1].plot(t, np.mean([[r['change_in_state_mse'] for r in c['velocity']] for c in rows], axis=0),
                       'o-', color=cl, markersize=4)
    for ax, title, ylabel in zip(axs.flat, ['E | State MAE', 'E | Signed intensity bias',
                                           'F | Teacher versus free velocity', 'F | Next step change in state MSE'],
                                ['MAE to ideal', 'Free − ideal', 'Velocity MSE', 'MSE increment']):
        ax.set(xlabel='Time t', title=title, ylabel=ylabel)
        ax.axhline(0, color='#666666', linewidth=.6)
        ax.grid(alpha=.18)
    axs[0, 0].legend(fontsize=8)
    axs[1, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f'FM stability | Matched patients: {len(next(iter(groups.values())))}')
    render(fig, 'trajectory.png')


def heatmaps(core, first):
    rows = [r for r in core if r['case'] == first]
    if not rows:
        return
    view = ROOT/'reports/truth'/first/'view.npz'
    with np.load(view) as labels:
        faz = labels['mask_faz']
        indices = np.argwhere(faz)
        focus = tuple(slice(max(0, i-48), min(304, i+49)) for i in indices.mean(0).astype(int)) if len(indices) else (slice(104, 201),)*2
    fig, axes = plt.subplots(5, len(rows), figsize=(3.2*len(rows), 12), squeeze=False, constrained_layout=True)
    for j, r in enumerate(rows):
        with np.load(ROOT/'reports/core'/r['model']/first/'signed_error.npz') as maps:
            for i, name in enumerate(['Bscan', 'Orthogonal', 'ILM_OPL', 'OPL_BM', 'FAZ crop']):
                a = maps['ILM_OPL'][focus] if name == 'FAZ crop' else maps[name]
                ax = axes[i, j]
                color = ax.imshow(a, cmap='RdBu_r', vmin=-.12, vmax=.12, interpolation='nearest')
                if name == 'FAZ crop' and faz[focus].any():
                    ax.contour(faz[focus], [.5], colors=['#26223b'], linewidths=.6)
                ax.axis('off')
                ax.set_title((r['model']+' | ' if i == 0 else '')+name, fontsize=9)
    fig.colorbar(color, ax=axes, shrink=.55, label='Prediction − GT / [0, 1]', extend='both')
    fig.suptitle(f'C | {first} signed errors, matching display scale')
    render(fig, 'signed_error_first.png')
    for r in rows:
        if not r['trajectory']:
            continue
        fig, axes = plt.subplots(4, 9, figsize=(19, 8.5), constrained_layout=True)
        folder = ROOT/'reports/core'/r['model']/first/'trajectory'
        for step in range(9):
            with np.load(folder/f'step_{step:02d}.npz') as arrays:
                for i, key in enumerate(['ideal_Bscan', 'free_Bscan', 'ideal_ILM_OPL', 'free_ILM_OPL']):
                    ax = axes[i, step]
                    ax.imshow(arrays[key], cmap='gray', vmin=0, vmax=.5, interpolation='nearest')
                    ax.axis('off')
                    if i == 0:
                        ax.set_title(f't={step/8:.3f}', fontsize=9)
                    if step == 0:
                        ax.set_title(key.replace('_', ' ') + (f' t={step/8:.3f}' if i == 0 else ''), fontsize=9)
        fig.suptitle(f'E | {r["model"]} / {first} | Ideal and free states at the same t')
        render(fig, f'trajectory_{r["model"]}_{first}.png')


def extended_figures(solvers, perturbation):
    if solvers:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
        for tag in sorted({r['model'] for r in solvers}):
            rows = [r for r in solvers if r['model'] == tag]
            expected = {f'Euler{n}' for n in (1, 2, 4, 8, 16, 32)} | {'Heun4', 'Heun8'}
            by_case = defaultdict(set)
            for r in rows:
                by_case[r['case']].add(r['solver']+str(r['steps']))
            complete = {c for c, v in by_case.items() if v == expected}
            if not complete:
                continue
            for solver in ('Euler', 'Heun'):
                selected = [r for r in rows if r['case'] in complete and r['solver'] == solver]
                for ax, key in zip(axs, ('ssim', 'bias')):
                    steps = sorted({r['steps'] for r in selected})
                    y = [np.mean([r['standard']['volume'][key] if key == 'ssim' else r['raw']['bias']
                                  for r in selected if r['steps'] == n]) for n in steps]
                    ax.plot(np.asarray(steps)*(1 if solver == 'Euler' else 2), y, 'o-' if solver == 'Euler' else 's--',
                            color=COLORS[tag], label=f'{tag} {solver} (n={len(complete)})')
        for ax, key in zip(axs, ('Volume SSIM', 'Raw intensity bias')):
            ax.set(xlabel='Full-volume velocity evaluations (NFE)', ylabel=key, xscale='log', title=key)
            ax.grid(alpha=.2)
        if axs[0].has_data():
            axs[0].legend(fontsize=7)
            fig.suptitle('H | Integrator comparison, complete paired solver sets')
            render(fig, 'solvers.png')
        else:
            plt.close(fig)
    data = [r for r in perturbation if r.get('mae_ratio') is not None]
    if data:
        order = ['offset_+0.005', 'offset_+0.01', 'offset_+0.02', 'offset_-0.02', 'gaussian', 'small_vessel_attenuation']
        fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for ax, t in zip(axs, (.25, .5, .75)):
            for tag in sorted({r['model'] for r in data}):
                values = []
                for key in order:
                    subset = [r['mae_ratio'] for r in data if r['model'] == tag and r['start_t'] == t and r['perturbation'] == key]
                    values.append(np.mean(subset) if subset else np.nan)
                ax.plot(np.arange(6), values, 'o-', color=COLORS[tag], label=tag)
            ax.axhline(1, c='#555555', lw=.8, ls='--')
            ax.set_title(f't={t:g}')
            ax.set_xticks(np.arange(6), ['+.005', '+.01', '+.02', '−.02', 'Noise', 'Small\nvessel'])
            ax.grid(alpha=.2)
        axs[0].set_ylabel('Final MAE to control / initial perturbation MAE')
        axs[0].legend(fontsize=7)
        fig.suptitle('G | Perturbation persistence vs the matched control')
        render(fig, 'perturbation.png')


def verify_complete(cfg, core, solver, perturb):
    """Validate the requested cells, preserved full volumes and receipt hashes."""
    models = set(cfg['models'])
    fm = models - {'XOCT_E213'}
    patients = set(cfg['validation_cases'])
    assert {(r['model'], r['case']) for r in core} == {(m, c) for m in models for c in patients}
    for r in core:
        if r['model'] in fm:
            assert len(r['trajectory']) == 9 and len(r['velocity']) == 8
            assert sum('/step_' in key and key.endswith('.npy') for key in r['files']) == 9
        assert set(r['endpoint']['standard']) == set(OBJECTS)
    expected_solver = {(m, c, solver, n) for m in fm for c in patients
                       for solver, sizes in cfg['solver_grid'].items() for n in sizes}
    assert {(r['model'], r['case'], r['solver'], r['steps']) for r in solver} == expected_solver
    expected_perturb = {(m, c, t, p) for m in fm for c in patients for t in cfg['perturb_steps']
                        for p in ('control', 'gaussian', 'small_vessel_attenuation', *[
                            f'offset_{x:+g}' for x in cfg['offsets']])}
    assert {(r['model'], r['case'], r['start_step'], r['perturbation']) for r in perturb} == expected_perturb
    count = 0
    for row in [*core, *solver, *perturb, *collect('reports/truth/*/record.json')]:
        for relative, expected in row['files'].items():
            file = (ROOT/relative).resolve()
            assert file.is_relative_to(ROOT)
            with file.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, relative
            if relative.startswith('predictions/trajectory/'):
                volume = np.load(file, mmap_mode='r', allow_pickle=False)
                assert list(volume.shape) == cfg['shape'] and volume.dtype == np.float32
            count += 1
    return count


def main(require_complete=False):
    cfg = load(ROOT/'experiment.json')
    core = collect('reports/core/*/*/record.json')
    solvers = collect('reports/extended/*/*/solver/*/record.json')
    perturb = collect('reports/extended/*/*/perturb/*/*/record.json')
    OUT.mkdir(exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    standard, regional, distributions, depth, track, velocities, extras = [], [], [], [], [], [], []
    for row in core:
        label = dict(model=row['model'], case=row['case'])
        score_rows(standard, row['endpoint'], row, 'unadjusted')
        for name, result in row['oracle'].items():
            score_rows(standard, result, row, 'GT_oracle_'+name)
        extras.append(dict(**label, **row['topology']))
        for kind in ('raw', 'clipped'):
            values = row[kind]
            d = values['distribution']
            distributions.append(dict(**label, stage=kind, **{k: v for k, v in d.items() if k != 'histogram'}))
            regional += [dict(**label, stage=kind, region=name, **stats) for name, stats in values['regions'].items()]
            for z, entries in enumerate(zip(*values['by_depth'].values())):
                depth.append(dict(**label, stage=kind, z=z, **dict(zip(values['by_depth'], entries))))
        for state in row['trajectory']:
            track.append(dict(**label, **{k: v for k, v in state.items() if k != 'clipped'},
                              **{f'clipped_{k}': v for k, v in state['clipped'].items()}))
        velocities += [dict(**label, **r) for r in row['velocity']]
    solver_rows, recovery_rows = [], []
    for row in solvers:
        treatment = row['solver']+str(row['steps'])
        score_rows(standard, row, row, treatment)
        solver_rows.append(dict(model=row['model'], case=row['case'], solver=row['solver'], steps=row['steps'],
                                nfe=row['nfe'], physical_new_calls=row['physical_new_calls'],
                                seconds=row['seconds'], **{f'raw_{key}': v for key, v in row['raw'].items()}))
    for row in perturb:
        treatment = f"G_t{row['start_t']:g}_{row['perturbation']}"
        score_rows(standard, row, row, treatment)
        names = ['model', 'case', 'start_t', 'perturbation', 'nfe', 'mae_ratio', 'signed_bias_ratio', 'seed']
        recovery_rows.append(dict(**{key: row.get(key) for key in names},
                                  **{f'{group}_{key}': v for group in ['versus_control_initial',
                                     'versus_control_final', 'versus_control_clipped', 'raw']
                                     for key, v in row.get(group, {}).items()}))
    standard_means = means(standard, ('model', 'treatment', 'object'), METRICS)
    for name, rows in [('per_patient_standard', standard), ('standard_means', standard_means),
                       ('intensity_distributions', distributions), ('gt_region_bias', regional),
                       ('depth_profiles', depth), ('trajectory', track), ('velocity_teacher_free', velocities),
                       ('perturbation_recovery', recovery_rows), ('solver_runtime', solver_rows),
                       ('topology', extras)]:
        table(OUT/f'{name}.csv', rows)
    curves(core)
    heatmaps(core, cfg['first_case'])
    extended_figures(solvers, perturb)
    checked = verify_complete(cfg, core, solvers, perturb) if require_complete else None
    counts = dict(core=len(core), solver=len(solvers), perturbation=len(perturb))
    expected = dict(core=50, solver=320, perturbation=840)
    summary = dict(counts=counts, expected=expected, verified_files=checked, complete=checked is not None)
    (OUT/'status.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    lines = ['# exp182 · FM 强度与轨迹联合诊断', '',
             f"已提交核心 {len(core)}/50 个模型病例，求解器 {len(solvers)}/320 项，扰动及对照 {len(perturb)}/840 项。",
             '各组均值旁列出实际患者数。覆盖不同时使用逐患者表匹配病例后再比较。', '',
             '固定验证集，完整 256×304×304，强度固定除 255，指标范围 1。SSIM 采用均匀窗口 7 和样本协方差。'
             '逐 B-scan 覆盖全部 304 张 [Z,X] 切片，先卷内平均再对患者等权。投影使用相同 GT 分层。',
             'E125 历史快照没有可用检查点。本次以可用的 E90 作低谷诊断，E120/E130 为原方案权重，'
             'E190 与 XOCT E213 为各自原选模参考。', '',
             '| 权重 | 患者 | 对象 | MAE ↓ | PSNR ↑ | SSIM ↑ |',
             '|---|---:|---|---:|---:|---:|']
    for r in standard_means:
        if r['treatment'] == 'unadjusted':
            lines.append(f"| {r['model']} | {r['patients']} | {r['object']} | {r['mae']:.6f} | {r['psnr']:.3f} | {r['ssim']:.6f} |")
    oracle = oracle_summary(core)
    table(OUT/'oracle_volume_delta.csv', oracle)
    if oracle:
        lines += ['', '### 强度校正敏感性（D）', '',
                  '每条变化先在同一个病例和同一个权重内计算，再按患者等权平均。对照为未校正八步终点或XOCT直接输出。',
                  '校正使用GT信息，所有分支均在评价前截断到[0,1]。', '',
                  '| 权重 | 病例数 | GT校正 | Δ3D MAE | Δ3D PSNR | Δ3D SSIM |',
                  '|---|---:|---|---:|---:|---:|']
        for r in oracle:
            lines.append(f"| {r['model']} | {r['patients']} | {r['correction']} | "
                         f"{r['delta_mae']:+.6f} | {r['delta_psnr']:+.3f} | {r['delta_ssim']:+.6f} |")
    if core:
        lines += ['', '### 轨迹与速度（E/F）', '',
                  '偏差及MSE均以相同时间的理想状态为参考。表中两个时刻用于快速定位漂移，完整九步结果见 trajectory.csv。'
                  '末次速度比较来自 t=0.875，误差增减需同时查看下一步状态MSE。', '',
                  '| 权重 | 病例数 | t=0.125偏差 | t=0.5偏差 | t=1偏差 | t=0.875 free−teacher速度MSE |',
                  '|---|---:|---:|---:|---:|---:|']
        for tag in cfg['models']:
            selected = [r for r in core if r['model'] == tag and r['trajectory']]
            if selected:
                shifts = [np.mean([r['trajectory'][i]['bias'] for r in selected]) for i in (1, 4, 8)]
                gap = np.mean([r['velocity'][-1]['teacher_free_gap'] for r in selected])
                lines.append(f"| {tag} | {len(selected)} | {shifts[0]:+.6f} | {shifts[1]:+.6f} | "
                             f"{shifts[2]:+.6f} | {gap:+.6f} |")
    lines += ['', '### 诊断证据', '',
              '- A/B：GT 区域和逐深度偏差见 gt_region_bias.csv 与 depth_profiles.csv。强度分位数、范围和越界率见 intensity_distributions.csv。',
              '- C：固定首例的有符号误差图使用同一色标。全部病例相同切面和投影的数值保留在 reports/core。',
              '- D：standard_means.csv 中 GT_oracle 为利用同一病例 GT 的敏感性结果，仅用于分析强度贡献。',
              '- E/F：trajectory.csv 对比相同时间的理想路径。velocity_teacher_free.csv 同时记录速度 MSE 差和下一步状态误差变化，区分偏离目标速度与主动抵消已有误差。',
              '- G：perturbation_recovery.csv 中 MAE 比低于 1 代表扰动相对配对对照缩小，高于 1 代表放大。补充保留全局亮度偏移的有符号比例。',
              '- H：solver_runtime.csv 记录步长、NFE、复用与耗时。standard_means.csv 提供各求解器完整三指标。',
              '']
    for image in sorted((OUT/'figures').glob('*.png')):
        lines.append(f'![{image.stem}](figures/{image.name})\n')
    (OUT/'summary.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    main(parser.parse_args().require_complete)

"""Patient-matched comparison using each arm's predeclared validation-selected checkpoint."""

import argparse
import csv
import html
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import common as c


OBJECTS = (('volume', 'Full 3D volume'), ('bscan2d', 'B-scan 2D average'),
           ('projection_ilm_opl', 'ILM-OPL max'), ('projection_opl_bm', 'OPL-BM max'),
           ('projection_ilm_opl_mean', 'ILM-OPL mean'), ('projection_opl_bm_mean', 'OPL-BM mean'),
           ('projection_ilm_bm_mean', 'ILM-BM mean'), ('projection_full', 'Full-Z mean'),
           ('projection_zmax', 'Full-Z max'))


def load(split):
    roots = [c.ROOT.parent/name for name in c.read(c.ROOT/'experiment.json')['suite_experiments']]
    reports, folders = [], []
    for root in roots:
        state = c.read(root/'runs/train/status.json')
        selected = state.get('best')
        if not selected:
            raise ValueError('No committed validation-best checkpoint yet: '+root.name)
        if split == 'validation':
            folder = root/selected['report']
        else:
            complete = c.read(root/'runs/test_complete.json')
            if complete['epoch'] != selected['epoch']:
                raise ValueError('Formal test does not use this arm\'s declared validation best')
            folder = root/complete['report']
        report = c.read(folder/'metrics.json')
        if report['model_tensor_sha256'] != selected['model_tensor_sha256'] or report['split'] != split:
            raise ValueError('Comparison checkpoint identity mismatch')
        reports.append(report)
        folders.append(folder)
    if c.read(roots[0]/'membership.json') != c.read(roots[1]/'membership.json'):
        raise ValueError('Patient membership differs between arms')
    indexed = [{row['case_id']: row for row in report['cases']} for report in reports]
    if set(indexed[0]) != set(indexed[1]):
        raise ValueError('Paired comparison needs complete identical patients')
    for case in indexed[0]:
        if indexed[0][case]['input_sha256'] != indexed[1][case]['input_sha256']:
            raise ValueError('Paired comparison input changed')
    return roots, folders, reports, indexed


def build(split):
    roots, folders, reports, indexed = load(split)
    out = roots[0]/'reports'/('comparison_'+split)
    out.mkdir(parents=True, exist_ok=True)
    (out/'cases').mkdir(exist_ok=True)
    labels = ['A control E'+str(reports[0]['epoch']), 'B layer means E'+str(reports[1]['epoch'])]
    rows, table = [], []
    for key, title in OBJECTS:
        for metric in ('mae', 'psnr', 'ssim'):
            a, b = [report['metrics'][key+'_'+metric] for report in reports]
            best = min(a, b) if metric == 'mae' else max(a, b)
            cells = [f'<strong>{value:.6f}</strong>' if value == best else f'{value:.6f}' for value in (a, b)]
            table.append(f'<tr><td>{title}</td><td>{metric.upper()}</td><td>{cells[0]}</td><td>{cells[1]}</td><td>{b-a:+.6f}</td></tr>')
            rows.append(dict(object=key, metric=metric, control=a, layer_means=b, delta_B_minus_A=b-a))
    with (out/'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(3, 3, figsize=(14, 9.5), layout='constrained')
    for ax, (key, title) in zip(axes.flat, OBJECTS):
        values = [report['metrics'][key+'_ssim'] for report in reports]
        ax.bar((0, 1), values, color=('#47759b', '#2f8973'), width=.6)
        for index, value in enumerate(values):
            ax.text(index, value, f'{value:.4f}', ha='center', va='bottom', fontweight='bold' if value == max(values) else 'normal')
        ax.set(title=title, xticks=(0, 1), xticklabels=('A', 'B'), ylim=(0, max(values)*1.18), ylabel='SSIM')
    fig.suptitle('Same patients, original four-projection validation selection', fontsize=14)
    for extension in ('png', 'svg'):
        fig.savefig(out/('ssim_comparison.'+extension), dpi=150, facecolor='white')
    plt.close(fig)
    cards = []
    display = (('ILM_OPL', 'ILM-OPL max', .5), ('OPL_BM', 'OPL-BM max', .5),
               ('ILM_OPL_MEAN', 'ILM-OPL mean', .12), ('OPL_BM_MEAN', 'OPL-BM mean', .12),
               ('ILM_BM_MEAN', 'ILM-BM mean', .12))
    for case in sorted(indexed[0]):
        with np.load(folders[0]/'views'/(case+'.npz'), allow_pickle=False) as a:
            left = {key: a[key] for key in a.files}
        with np.load(folders[1]/'views'/(case+'.npz'), allow_pickle=False) as b:
            right = {key: b[key] for key in b.files}
        fig, axes = plt.subplots(5, 3, figsize=(10, 15), layout='constrained')
        for index, (key, title, maximum) in enumerate(display):
            if not np.array_equal(left['gt_'+key], right['gt_'+key]):
                raise ValueError('Paired figure GT projection differs')
            for column, array in enumerate((left['gt_'+key], left['pred_'+key], right['pred_'+key])):
                ax = axes[index, column]
                ax.imshow(array, cmap='gray', vmin=0, vmax=maximum, interpolation='nearest')
                ax.set(xticks=[], yticks=[])
                if index == 0:
                    ax.set_title(('Ground truth', *labels)[column])
                if column == 0:
                    ax.set_ylabel(title)
        fig.suptitle(case+' | same GT and fixed display windows', fontsize=14)
        fig.savefig(out/'cases'/(case+'.png'), dpi=130, facecolor='white')
        plt.close(fig)
        cards.append(f'<a href="cases/{case}.png"><img loading="lazy" src="cases/{case}.png" alt="{case}"></a>')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Layer projection comparison</title>'
        '<style>body{font:16px/1.7 sans-serif;max-width:1200px;margin:30px auto;padding:20px}table{border-collapse:collapse;width:100%}'
        'td,th{border-bottom:1px solid #ddd;padding:8px;text-align:right}td:first-child{text-align:left}img{width:100%;height:auto}</style>'
        f'<h1>A / B · {html.escape(split)} · {reports[0]["patients"]} patients</h1>'
        '<p>各组按浅层最大投影与三张层内均值投影 SSIM 的等权平均选择权重。全部指标和图像来自各组同一权重。</p>'
        '<p>MAE 范围 0–1，PSNR 单位 dB，SSIM 使用固定范围1与win7。完整3D、逐B-scan和七种投影分别评分，每患者等权。</p>'
        '<p>'+html.escape(' / '.join(labels))+'</p><img src="ssim_comparison.png">'
        '<table><tr><th>Object</th><th>Metric</th><th>A</th><th>B</th><th>B−A</th></tr>'+''.join(table)+'</table>'
        '<p><a href="metrics.csv">完整数值CSV</a></p>'+''.join(cards)+'</html>')
    (out/'index.html').write_text(document, encoding='utf-8')
    c.write(out/'identity.json', dict(split=split, roots=[str(root) for root in roots],
            epochs=[report['epoch'] for report in reports], model_digests=[report['model_tensor_sha256'] for report in reports],
            patients=sorted(indexed[0]), source_reports=[str(folder) for folder in folders]))
    print('Comparison '+str(out/'index.html'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('validation', 'test'), default='validation')
    build(parser.parse_args().split)

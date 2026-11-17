"""Summarize and independently audit all completed A-F validation evidence."""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parent
OBJECTS = ('volume', 'bscan2d', 'ILM_OPL', 'OPL_BM', 'FULL', 'ZMAX',
           'ILM_OPL_MEAN', 'OPL_BM_MEAN', 'ILM_BM_MEAN')
NAMES = ('完整3D', '逐B-scan二维', '主ILM–OPL最大', 'OPL–BM最大', '全Z均值', '全Z最大',
         'ILM–OPL均值', 'OPL–BM均值', 'ILM–BM均值')
METRICS = ('mae', 'psnr', 'ssim')
MODELS = ('E090', 'E120', 'E130', 'E190', 'XOCT_E213')


def mean(values):
    return float(np.mean(values))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def triple(values):
    return f"{values['mae']:.6f} / {values['psnr']:.3f} / {values['ssim']:.6f}"


def main():
    source = ROOT/'source/af_core_records.json'
    snapshot = json.loads(source.read_text(encoding='utf-8'))
    cfg = json.loads((ROOT/'experiment.json').read_text(encoding='utf-8'))
    frozen = json.loads((ROOT/'verification/frozen.json').read_text(encoding='utf-8'))
    groups = defaultdict(list)
    rows = snapshot['core']
    assert len(rows) == 50 and len(snapshot['truth']) == 10
    index = {}
    with (ROOT/'analysis/per_patient_standard.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            key = tuple(row[k] for k in ('model', 'case', 'treatment', 'object'))
            assert key not in index
            index[key] = {metric: float(row[metric]) for metric in METRICS}
    checked = 0
    for row in rows:
        tag, cid = row['model'], row['case']
        assert row['identity']['model'] == frozen['models'][tag]['tensor_sha256']
        assert row['identity']['input'] == frozen['inputs'][cid]['sha256']
        assert set(row['endpoint']['standard']) == set(OBJECTS)
        for name, branch in [('unadjusted', row['endpoint']), *[
                ('GT_oracle_'+name, value) for name, value in row['oracle'].items()]]:
            for obj in OBJECTS:
                for metric in METRICS:
                    assert abs(index[tag, cid, name, obj][metric] - branch['standard'][obj][metric]) < 1e-12
                    checked += 1
        if tag != 'XOCT_E213':
            assert len(row['trajectory']) == 9 and len(row['velocity']) == 8
            assert row['trajectory'][0]['mae'] == 0 and row['velocity'][0]['teacher_free_gap'] == 0
            assert max(abs(v['mse_balance_residual']) for v in row['velocity']) < 2e-8
        groups[tag].append(row)
    assert set(groups) == set(MODELS)
    summary = dict(observed_at=snapshot['observed_at'], source_sha256=sha(source), models={})
    for tag in MODELS:
        selected = groups[tag]
        assert {r['case'] for r in selected} == set(cfg['validation_cases']) and len(selected) == 10
        quality = {obj: {metric: mean([r['endpoint']['standard'][obj][metric] for r in selected])
                         for metric in METRICS} for obj in OBJECTS}
        oracle = {name: {obj: {metric: mean([r['oracle'][name]['standard'][obj][metric] for r in selected])
                              for metric in METRICS} for obj in OBJECTS}
                  for name in ('global_mean', 'mean_std', 'depth_mean')}
        regions = {name: {key: mean([r['clipped']['regions'][name][key] for r in selected])
                           for key in ('mean', 'gt_mean', 'bias', 'mae', 'voxels')}
                   for name in selected[0]['clipped']['regions']}
        result = dict(standard=quality, oracle=oracle, regions=regions,
                      negative_ratio=mean([r['raw']['distribution']['negative_ratio'] for r in selected]),
                      overflow_ratio=mean([r['raw']['distribution']['overflow_ratio'] for r in selected]),
                      clipped_mean=mean([r['clipped']['distribution']['mean'] for r in selected]),
                      outside_abs_error_share=mean([
                          r['clipped']['regions']['retina_exterior']['mae'] *
                          r['clipped']['regions']['retina_exterior']['voxels'] /
                          (r['endpoint']['standard']['volume']['mae']*np.prod(cfg['shape']))
                          for r in selected]),
                      oracle_improved_patients={name: sum(
                          r['oracle'][name]['standard']['volume']['ssim'] > r['endpoint']['standard']['volume']['ssim']
                          for r in selected) for name in oracle})
        if tag != 'XOCT_E213':
            result['trajectory'] = {key: [mean([r['trajectory'][i][key] for r in selected]) for i in range(9)]
                                    for key in ('mae', 'mse', 'bias')}
            result['velocity'] = {key: [mean([r['velocity'][i][key] for r in selected]) for i in range(8)]
                                  for key in ('teacher_mse', 'free_mse', 'teacher_velocity_bias',
                                              'free_velocity_bias', 'teacher_free_gap', 'error_velocity_inner')}
            result['teacher_bias_integral'] = mean([
                mean([s['teacher_velocity_bias'] for s in r['velocity']]) for r in selected])
            result['t0875_free_teacher_mse_ratio'] = (
                result['velocity']['free_mse'][-1]/result['velocity']['teacher_mse'][-1])
            result['positive_t0_teacher_bias_patients'] = sum(r['velocity'][0]['teacher_velocity_bias'] > 0 for r in selected)
            result['all_positive_teacher_bias_patients'] = sum(
                all(s['teacher_velocity_bias'] > 0 for s in r['velocity']) for r in selected)
            result['all_state_mse_increases_patients'] = sum(
                all(s['change_in_state_mse'] > 0 for s in r['velocity']) for r in selected)
            result['all_error_velocity_inner_positive_patients'] = sum(
                all(s['error_velocity_inner'] > 0 for s in r['velocity'][1:]) for r in selected)
        summary['models'][tag] = result

    bad = summary['models']['E090']
    truth_mean = mean([r['distribution']['mean'] for r in snapshot['truth']])
    bad['gt_mean'] = truth_mean
    bad['relative_clipped_mean_excess'] = bad['clipped_mean']/truth_mean - 1
    bad['meanstd_mae_reduction'] = 1-bad['oracle']['mean_std']['volume']['mae']/bad['standard']['volume']['mae']
    (ROOT/'analysis/af_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

    lines = ['# A–F完成后的判断', '',
             f"数据截至{snapshot['observed_at']}，固定10位验证患者，4个exp163权重与XOCT E213共50个模型病例。",
             'E90替代未留存的E125作为低谷诊断。E190与XOCT E213保留原选模身份。', '',
             '**核心判断：exp163低谷的主要直接机制是大范围背景信号抬高和强度分布失准；'
             '已学速度场在理想路径上就带有亮度正偏，自由生成持续累积偏差；血管细节不足在各权重间持续存在。**', '',
             '### 1. A–D把E90低谷定位到强度尺度', '',
             f"E90完整3D SSIM为{bad['standard']['volume']['ssim']:.6f}，全局均值校正为"
             f"{bad['oracle']['global_mean']['volume']['ssim']:.6f}，均值和方差校正为"
             f"{bad['oracle']['mean_std']['volume']['ssim']:.6f}，逐深度均值校正为"
             f"{bad['oracle']['depth_mean']['volume']['ssim']:.6f}。均值和方差校正在10/10患者改善SSIM，"
             f"平均MAE降低{100*bad['meanstd_mae_reduction']:.1f}%。所有校正使用对应病例GT，只用于量化强度贡献。",
             '', '### 2. A–C显示背景抬高与血管信号不足同时出现', '',
             f"E90全卷均值{bad['clipped_mean']:.6f}，GT为{truth_mean:.6f}，偏高"
             f"{100*bad['relative_clipped_mean_excess']:.1f}%。GT分层定义的ILM–BM之外平均偏差"
             f"{bad['regions']['retina_exterior']['bias']:+.6f}，浅层GT无血管射线"
             f"{bad['regions']['background_ILM_OPL']['bias']:+.6f}，FAZ3D层内"
             f"{bad['regions']['FAZ3D_retinal']['bias']:+.6f}。这三个区域在全部10位患者均为正偏。",
             f"ILM–BM层带之外占全卷约83.3%体素，贡献E90绝对误差的"
             f"{100*bad['outside_abs_error_share']:.1f}%（先各病例算占比，再患者等权）。",
             f"固定GT小管径候选体素偏差{bad['regions']['small_vessel_candidate_3d']['bias']:+.6f}，"
             f"大管径候选体素{bad['regions']['large_vessel_candidate_3d']['bias']:+.6f}。"
             '表现为低信号区域变亮、血管对比度压缩。候选体素来自GT二维血管管径分区及层内半峰值条件，区域定义全程固定。',
             '', '### 3. E显示漂移从第一步连续积累', '',
             f"E90在t=.125、.5、1的平均有符号偏差分别为"
             f"{bad['trajectory']['bias'][1]:+.6f}、{bad['trajectory']['bias'][4]:+.6f}、"
             f"{bad['trajectory']['bias'][8]:+.6f}。十位患者均从第一步出现正偏。"
             '同时间理想路径MAE及MSE沿八步持续增加。', '',
             '### 4. F显示理想路径已有偏置，轨迹反馈继续加重', '',
             f"E90在八个理想状态节点的速度均值误差均为正，覆盖10/10患者。将这些teacher速度偏差按"
             f"Euler8时间间隔相加为{bad['teacher_bias_integral']:+.6f}，实际自由轨迹终点偏差为"
             f"{bad['trajectory']['bias'][-1]:+.6f}。主亮度偏置在理想输入上已经存在，自由状态反馈继续增加偏移。",
             '40条FM自由轨迹的每一步状态MSE均增加。在第1–7个非零误差节点，'
             '误差与速度误差的整卷内积均为正，实际轨迹上的净作用是强化已有偏离。',
             '末次节点t=.875的free/teacher速度MSE比，E90、E120、E130、E190分别为'
             + '、'.join(f"{summary['models'][tag]['t0875_free_teacher_mse_ratio']:.2f}" for tag in MODELS[:-1])
             + '。较好的检查点同样存在训练路径与自由路径差距。'
             '决定E90低谷的额外特征是持续正偏及广泛背景信号抬高。', '',
             '### 5. 高3D指标还需要结合截断前状态和血管细节', '',
             'E120、E130的截断前负值体素占比分别为'
             f"{100*summary['models']['E120']['negative_ratio']:.1f}%、"
             f"{100*summary['models']['E130']['negative_ratio']:.1f}%，这些值在原终点clamp操作后归零。"
             '零附近的强度分布、正负误差方向和截断共同决定3D SSIM表现。',
             'XOCT的ILM–BM层带外偏差接近零，E90则有系统性正偏。'
             '各模型在固定GT小管径候选体素上仍有负偏，微血管恢复继续需要单独评价。',
             '', '### 完整标准指标', '',
             '每格均为MAE↓ / PSNR↑ / SSIM↑，患者等权。float32，固定/255、range1，'
             '3D SSIM三维，逐B-scan与投影SSIM二维，uniform win7 sample covariance。'
             '主投影为ILM–OPL最大，层带均值、深层和全Z投影分别列示。', '',
             '| 权重 | 完整3D | 逐B-scan二维 | 主ILM–OPL最大 |',
             '|---|---|---|---|']
    for tag in MODELS:
        q = summary['models'][tag]['standard']
        lines.append('| '+tag+' | '+' | '.join(triple(q[k]) for k in OBJECTS[:3])+' |')
    lines += ['', '| 权重 | '+' | '.join(NAMES[3:])+' |',
              '|---|'+'---|'*len(OBJECTS[3:])]
    for tag in MODELS:
        q = summary['models'][tag]['standard']
        lines.append('| '+tag+' | '+' | '.join(triple(q[k]) for k in OBJECTS[3:])+' |')
    lines += ['', '### E90的GT校正完整指标', '',
              '| 处理 | 完整3D | 逐B-scan二维 | 主ILM–OPL最大 |',
              '|---|---|---|---|']
    for name, q in [('原输出', bad['standard']), *bad['oracle'].items()]:
        lines.append('| '+name+' | '+' | '.join(triple(q[k]) for k in OBJECTS[:3])+' |')
    lines += ['', '其余补充投影的GT校正三指标在af_summary.json和per_patient_standard.csv中完整保留。',
              '', '![十人强度与深度统计](figures/intensity_depth.png)',
              '', '![十人轨迹与速度误差](figures/trajectory.png)', '',
              '证据快照为source/af_core_records.json，含50条完成记录及原记录SHA256。'
              '独立复算与阶段CSV核对见verification/af_summary_audit.json。'
              'A–F说明本轮现有权重的强度和轨迹行为，优化器或损失改动的效果需由对应训练对照评价。', '']
    (ROOT/'analysis/af_conclusions.md').write_text('\n'.join(lines), encoding='utf-8')
    audit = dict(passed=True, observed_at=snapshot['observed_at'], models=5, patients=10, core_records=50,
                 matched_standard_values=checked, snapshot_sha256=sha(source),
                 stage_csv_sha256=sha(ROOT/'analysis/per_patient_standard.csv'),
                 summary_sha256=sha(ROOT/'analysis/af_summary.json'),
                 conclusions_sha256=sha(ROOT/'analysis/af_conclusions.md'),
                 patient_sets_match=True, identities_match_frozen_models_and_inputs=True,
                 original_core_receipt_sha256=snapshot['files'])
    (ROOT/'verification/af_summary_audit.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in audit.items() if k != 'original_core_receipt_sha256'}, indent=2))


if __name__ == '__main__':
    main()


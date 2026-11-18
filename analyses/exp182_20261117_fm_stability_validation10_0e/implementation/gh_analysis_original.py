"""Audit and summarize completed G/H records; no inference or training."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODELS = ('E090', 'E120', 'E130', 'E190')
SOLVERS = ('Euler1', 'Euler2', 'Euler4', 'Euler8', 'Euler16', 'Euler32', 'Heun4', 'Heun8')
OBJECTS = ('volume', 'bscan2d', 'ILM_OPL', 'OPL_BM', 'FULL', 'ZMAX',
           'ILM_OPL_MEAN', 'OPL_BM_MEAN', 'ILM_BM_MEAN')
LABELS = ('完整3D', '逐B-scan二维', '主ILM–OPL最大投影', 'OPL–BM最大投影',
          '全深度均值投影', '全深度最大投影', 'ILM–OPL层内均值',
          'OPL–BM层内均值', 'ILM–BM层内均值')
METRICS = ('mae', 'psnr', 'ssim')
PERTURBS = ('offset_+0.005', 'offset_+0.01', 'offset_+0.02', 'offset_-0.02',
            'gaussian', 'small_vessel_attenuation')
COLORS = dict(zip(MODELS, ('#c45337', '#3975a5', '#21856a', '#8654a6')))


def load(p):
    return json.loads(p.read_text(encoding='utf-8'))


def read_csv(p):
    with p.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stats(values, prefix):
    a = np.asarray(values, dtype=float)
    assert np.isfinite(a).all()
    return {prefix+'_mean':float(a.mean()), prefix+'_median':float(np.median(a)),
            prefix+'_min':float(a.min()), prefix+'_max':float(a.max()),
            prefix+'_gt1':int((a>1).sum()), prefix+'_lt1':int((a<1).sum())}


def main():
    source = ROOT/'source/gh_records.json'
    evidence = load(source)
    cfg = load(ROOT/'experiment.json')
    frozen = load(ROOT/'verification/frozen.json')
    assert hashlib.sha256((ROOT/'verification/frozen.json').read_bytes()).hexdigest() == evidence['frozen_sha256']
    patients = set(cfg['validation_cases'])
    solvers = evidence['groups']['solver']
    perturb = evidence['groups']['perturbation']
    assert evidence['status'] == load(ROOT/'analysis/status.json')
    assert evidence['status']['complete']
    assert len(solvers)==320 and len(perturb)==840
    assert {(r['model'],r['case'],r['solver']+str(r['steps'])) for r in solvers} == {
        (m,c,s) for m in MODELS for c in patients for s in SOLVERS}
    assert {(r['model'],r['case'],r['start_t'],r['perturbation']) for r in perturb} == {
        (m,c,t,s) for m in MODELS for c in patients for t in (.25,.5,.75) for s in ('control',*PERTURBS)}
    for r in [*solvers,*perturb]:
        assert r['identity'] == dict(model=frozen['models'][r['model']]['tensor_sha256'], input=frozen['inputs'][r['case']]['sha256'])
        assert set(r['standard']) == set(OBJECTS)

    retained = read_csv(ROOT/'analysis/per_patient_standard.csv')
    indexed = {(r['model'],r['case'],r['treatment'],r['object']):r for r in retained}
    assert len(indexed)==len(retained)==12240
    grouped = defaultdict(list)
    checked = 0
    for r in [*solvers,*perturb]:
        treatment = (r['solver']+str(r['steps']) if 'solver' in r else
                     f"G_t{r['start_t']:g}_{r['perturbation']}")
        for obj in OBJECTS:
            row = indexed[r['model'],r['case'],treatment,obj]
            for k in METRICS:
                assert abs(float(row[k])-r['standard'][obj][k]) < 1e-12
                checked += 1
            grouped[r['model'],treatment,obj].append(r['standard'][obj])
    means = {}
    summary_rows = []
    for (m,t,o), rows in grouped.items():
        assert len(rows)==10
        means[m,t,o] = {k:float(np.mean([r[k] for r in rows])) for k in METRICS}
        summary_rows.append(dict(model=m,treatment=t,object=o,patients=10,**means[m,t,o]))
    server_means = {(r['model'],r['treatment'],r['object']):r for r in read_csv(ROOT/'analysis/standard_means.csv')}
    for key, values in means.items():
        for k in METRICS:
            assert abs(values[k]-float(server_means[key][k])) < 1e-12
    write_csv('standard_metrics.csv', summary_rows)

    ggroups = []
    for m in MODELS:
        for t in (.25,.5,.75):
            for kind in PERTURBS:
                rows=[r for r in perturb if r['model']==m and r['start_t']==t and r['perturbation']==kind]
                initial=[r['versus_control_initial']['mae'] for r in rows]
                raw=[r['mae_ratio'] for r in rows]
                clipped=[r['versus_control_clipped']['mae']/i for r,i in zip(rows,initial)]
                assert all(abs(a-r['versus_control_final']['mae']/i)<1e-12 for r,a,i in zip(rows,raw,initial))
                ggroups.append(dict(model=m,start_t=t,perturbation=kind,patients=10,
                    initial_mae_mean=float(np.mean(initial)),**stats(raw,'raw_ratio'),**stats(clipped,'clipped_ratio'),
                    initial_bias_mean=float(np.mean([r['versus_control_initial']['bias'] for r in rows])),
                    final_bias_mean=float(np.mean([r['versus_control_final']['bias'] for r in rows]))))
    write_csv('recovery_groups.csv', ggroups)
    injected=[r for r in perturb if r['perturbation']!='control']
    all_recovery={kind:stats([r['mae_ratio'] for r in injected if r['perturbation']==kind],'raw_ratio')
                  for kind in PERTURBS}

    hrows=[]; pairs=[]
    lookup={(r['model'],r['case'],r['solver']+str(r['steps'])):r for r in solvers}
    for m in MODELS:
        for name in SOLVERS:
            rows=[lookup[m,c,name] for c in sorted(patients)]
            hrows.append(dict(model=m,solver=name,nfe=rows[0]['nfe'],patients=10,
                              raw_bias=float(np.mean([r['raw']['bias'] for r in rows])),
                              negative_ratio=float(np.mean([r['raw']['negative_ratio'] for r in rows])),
                              **{f'{o}_{k}':means[m,name,o][k] for o in OBJECTS for k in METRICS}))
            for obj in OBJECTS:
                for k in METRICS:
                    diff=np.array([r['standard'][obj][k]-lookup[m,r['case'],'Euler8']['standard'][obj][k] for r in rows])
                    pairs.append(dict(model=m,solver=name,reference='Euler8',object=obj,metric=k,patients=10,
                                      delta_mean=float(diff.mean()),delta_median=float(np.median(diff)),
                                      delta_min=float(diff.min()),delta_max=float(diff.max()),
                                      improved=int((diff<0 if k=='mae' else diff>0).sum()),worsened=int((diff>0 if k=='mae' else diff<0).sum())))
    write_csv('solver_means.csv',hrows)
    write_csv('solver_paired_deltas.csv',pairs)

    plt.rcParams.update({'font.family':FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                         'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(13.4,7.5),constrained_layout=True)
    kinds=('offset_+0.005','gaussian','small_vessel_attenuation')
    titles=('整体亮度 +0.005','高斯噪声 σ=0.01','细血管候选区域强度减半')
    for col,(kind,title) in enumerate(zip(kinds,titles)):
        for row,prefix in enumerate(('raw_ratio','clipped_ratio')):
            ax=axes[row,col]
            for m in MODELS:
                entries=[r for r in ggroups if r['model']==m and r['perturbation']==kind]
                x=np.array([r['start_t'] for r in entries]); y=np.array([r[prefix+'_mean'] for r in entries])
                lower=y-np.array([r[prefix+'_min'] for r in entries]); upper=np.array([r[prefix+'_max'] for r in entries])-y
                ax.errorbar(x,y,yerr=np.stack((lower,upper)),marker='o',capsize=2,lw=1.6,
                            label=m,color=COLORS[m],alpha=.9)
            ax.axhline(1,color='#555555',linestyle='--',lw=1)
            ax.set_xticks((.25,.5,.75)); ax.set_xlabel('扰动注入时刻 t')
            ax.set_title(title if row==0 else title+' · 最终截断后')
            ax.grid(alpha=.15)
            if col==0:ax.set_ylabel('终点相对配对对照 MAE / 初始扰动 MAE')
    axes[0,0].legend(ncol=2,fontsize=9)
    fig.suptitle('G：原始流场放大全部所测扰动；最终截断可压低部分可见误差\n10位验证患者等权均值，误差棒为患者最小至最大值；虚线1表示扰动大小不变',fontsize=13)
    fig.savefig(OUT/'g_recovery.png',dpi=170,facecolor='white');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13.4,4.1),constrained_layout=True)
    for ax,obj,title in zip(axes,('volume','bscan2d','ILM_OPL'),('完整3D SSIM','逐B-scan SSIM','主ILM–OPL最大投影 SSIM')):
        for m in MODELS:
            x=(1,2,4,8,16,32)
            ax.plot(x,[means[m,'Euler'+str(n),obj]['ssim'] for n in x],'-o',label=m,color=COLORS[m],ms=4)
            ax.plot((8,16),[means[m,'Heun'+str(n),obj]['ssim'] for n in (4,8)],'x--',color=COLORS[m],ms=7,alpha=.85)
        ax.set_xscale('log',base=2);ax.set_xticks((1,2,4,8,16,32),labels=('1','2','4','8','16','32'))
        ax.set_xlabel('速度场求值次数 NFE');ax.set_title(title);ax.grid(alpha=.15)
    axes[0].legend(fontsize=9)
    fig.suptitle('H：步数增加、换用Heun均未解决E90低谷\n实线圆点为Euler，虚线叉号为Heun；所有点均为固定10位验证患者均值',fontsize=13)
    fig.savefig(OUT/'h_solvers.png',dpi=170,facecolor='white');plt.close(fig)

    summary=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),observed=evidence['observed'],
        complete=True,standard_values_checked=checked,all_standard_groups=len(means),
        injected=len(injected),raw_amplified=sum(r['mae_ratio']>1 for r in injected),
        raw_min=min(r['mae_ratio'] for r in injected),raw_max=max(r['mae_ratio'] for r in injected),
        perturbation_summary=all_recovery,solver_summary=hrows)
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# exp182：G/H完整验证结果分析','',
           '分析与核对日期：2026年11月18日。实验建立/开始日期：2026年11月17日。',
           'A800于2026年11月18日00:06:07完成，核心与扩展进程均正常退出。最终报告校验2,220份文件。',
           '核心50/50、求解器320/320、扰动和无扰动对照840/840；G/H覆盖四个权重×固定十位验证患者。',
           '本次只读取结果并进行CPU汇总，没有训练、重新推理或改变原检查点选择。','',
           '**结论：所测扰动会被已学习的速度场放大；步长离散误差不能解释E90的严重强度低谷。** '
           '结合A–F，问题包含模型本身的强度偏差，以及沿自由轨迹继续传播、放大误差的能力缺陷。'
           'E130的高整卷SSIM没有消除这种缺陷，背景截断会影响其最终可见程度。','',
           '## G：模型会不会主动纠正扰动','',
           '起点为同一患者理想路径 $(1-t)X+tY$，t为0.25、0.5、0.75；加扰动与不加扰动各自继续剩余Euler8时间网格。'
           '本项不是在实际自由轨迹上注入，也不是以GT终点误差作为扰动幅度。比值分子为两条配对轨迹终点的MAE距离，分母为注入瞬间的MAE距离。',
           '840项包含120条无扰动对照及720条实际注入。720/720条原始未截断终点的比值均大于1。'
           f"全部实际注入的最小比值为{summary['raw_min']:.6f}，最大为{summary['raw_max']:.6f}。",'',
           '下表每格为原始终点的**放大倍数**，依次为 t=0.25 / 0.5 / 0.75，均为十人等权平均。','',
           '| 权重 | 亮度+0.005 | 高斯噪声σ=0.01 | 细血管候选区域减半 |',
           '|---|---:|---:|---:|']
    for m in MODELS:
        cells=[]
        for kind in kinds:
            rows=[r for r in ggroups if r['model']==m and r['perturbation']==kind]
            cells.append(' / '.join(f"{r['raw_ratio_mean']:.3f}" for r in rows))
        lines.append('| '+m+' | '+' | '.join(cells)+' |')
    lines += ['', '![G扰动恢复](g_recovery.png)','',
              '亮度的另外三档偏移（+0.01、+0.02、−0.02）同样全部被放大，详见[全部72组扰动汇总](recovery_groups.csv)。'
              '早期扰动经历更多剩余步骤，放大更明显。E130的原整卷SSIM为0.8370，但其早期亮度偏移放大1.184倍、'
              '高斯噪声1.534倍、细血管候选区域扰动5.046倍，表明高整卷分数不能代表误差收缩能力。','',
              '**最终截断需要单独看。** 以上结论来自未截断流场。将终点截断到[0,1]后，部分噪声和亮度扰动的可见比值小于1；'
              '例如E130晚期高斯噪声的均值为0.621。不能把这种结果直接归为网络主动恢复，截断本身会压掉负值。'
              '但细血管候选区域扰动的120/120次在最终截断后仍然放大。其整卷MAE距离包含局部扰动后向其他体素传播的影响，'
              '5倍不等于血管强度下降5倍；t较早时被减半的理想状态仍含较多OCT成分。','',
              '## H：增加步数或改求解器能否修复','',
              '原路线为Euler8；其余设置为诊断分支。同一权重、相同全分辨率、相同十人，最终统一截断后评价。'
              'Heun4与Euler8同为8次速度求值；Heun8与Euler16同为16次。Heun末步额外查询t=1，原训练随机t在[0,1)内，'
              '因此结果同时受方法与端点查询影响，不能据此断言所有高阶求解器都无效。','',
              'exp163原选模best为E190。各格按 **MAE / PSNR(dB) / SSIM** 排列。','',
              '| E190求解器 | NFE | 完整3D | 逐B-scan二维 | 主ILM–OPL最大投影 |',
              '|---|---:|---:|---:|---:|']
    fmt=lambda s:f"{s['mae']:.6f} / {s['psnr']:.3f} / {s['ssim']:.6f}"
    for name in SOLVERS:
        nfe=next(r['nfe'] for r in hrows if r['model']=='E190' and r['solver']==name)
        lines.append('| '+name+' | '+str(nfe)+' | '+' | '.join(fmt(means['E190',name,o]) for o in OBJECTS[:3])+' |')
    lines += ['', '![H求解器对照](h_solvers.png)','',
              '- E90：Euler8整卷SSIM为0.3646，Euler32为0.3867，Heun4为0.3692；任何受测分支都未恢复到0.8量级。'
              'Euler32原始全局偏差仍为+0.014660，接近Euler8的+0.015213。A–F中仅GT均值/标准差校正即达0.8492，'
              '因此严重低谷的首要原因是已学习输出的强度偏差，单纯积分精度不足无法解释。',
              '- E190：Euler1的整卷MAE下降12.52%、PSNR提高0.278dB、SSIM增加0.05558；'
              '主投影SSIM则由0.49043降至0.48297。一步与八步在体积指标和血管投影指标上存在取舍，不能仅凭整卷SSIM替换原路线。',
              '- 同8次求值：E190的Heun4整卷SSIM为0.75352，接近Euler8的0.75447；主投影SSIM仅增加0.00273。'
              '提高求解阶数没有解决强度稳定性问题。',
              '- 更多步数并非统一有益：E130由Euler8到Euler32整卷SSIM略增0.00137，'
              '但PSNR下降0.252dB、主投影SSIM下降0.00345。不能把“积分更细”直接当成“更接近GT”。','',
              '## 与A–F合并后的判断','',
              '1. E90的低谷主要由背景/整体强度偏差造成；G/H进一步说明这种偏差不能靠自由积分自行纠正，也不能靠加密步数根治。',
              '2. 四个权重均存在对受测扰动的非收缩行为，覆盖低分与高分权重；它是这组模型的共同轨迹问题，而非只有E90出现。',
              '3. 整卷强度稳定性与细血管重建质量需要分别验证。全局GT校正能诊断前者，却不增加新的血管空间信息；'
              'G的结构化区域扰动也没有被模型恢复。',
              '4. 下一轮科学实验应优先验证实际自由轨迹/终点监督及强度约束是否改善这些现象，并在相同原选模标准下保留全部指标。'
              '这是由诊断提出的后续方向，本次未创建或启动任何训练，也未证明某一种训练修改必然有效。','',
              '## 指标与证据范围','',
              '固定验证患者：'+', '.join(sorted(patients))+'。完整256×304×304、强度/255、data range=1。'
              '3D SSIM使用7³均匀窗及样本协方差；逐B-scan采用7×7窗，304张[Z,X]切片先卷内平均再患者等权；'
              '投影采用相同GT分层，二维7×7窗。投影含主ILM–OPL MIP、OPL–BM MIP、全深度均值、全深度最大值、'
              'ILM–OPL/OPL–BM/ILM–BM层内均值七组。区域扰动是GT二维标签和深度阈值形成的候选体素，非人工3D血管分割。',
              '四组FM均为同一exp163不同轮次，E190保留原四投影SSIM选出的best；E90/E120/E130为诊断权重。'
              '原历史E125权重缺失，以保留的E90作低谷诊断，二者不等同。'
              '张量float32，TF32 matmul/cuDNN允许，AMP/scaler关闭，benchmark开、deterministic关。'
              '本次没有使用50人测试集，也没有比较新训练策略。',
              'Euler8可能复用A–F终点，其他分支也复用t=0速度；日志耗时包含指标计算，且有并发负载差异，不作为公平速度结论。','',
              '- [全部G/H的1,044组MAE、PSNR、SSIM](standard_metrics.csv)',
              '- [四个权重×八种求解器的完整九对象指标](solver_means.csv)',
              '- [逐患者配对差的均值、中位数、范围及改善数量](solver_paired_deltas.csv)',
              '- [72组扰动传播汇总（原始与最终截断分开）](recovery_groups.csv)',
              '- 原始1,160条数值记录及40条组提交证据：[source/gh_records.json](../../source/gh_records.json)',
              '- A–F结论：[af_conclusions.md](../af_conclusions.md)',
              '- 本次校核：[verification.json](verification.json)','',
              '## H：各权重完整标准指标','']
    for m in MODELS:
        lines += ['### '+m,'','| 对象 | 求解器 | MAE↓ | PSNR(dB)↑ | SSIM↑ |','|---|---|---:|---:|---:|']
        for obj,label in zip(OBJECTS,LABELS):
            for name in SOLVERS:
                v=means[m,name,obj]
                lines.append(f"| {label} | {name} | {v['mae']:.6f} | {v['psnr']:.3f} | {v['ssim']:.6f} |")
        lines.append('')
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verification=dict(passed=True,source_sha256=summary['source_sha256'],
        frozen_sha256=evidence['frozen_sha256'],record_hashes=len(evidence['record_sha256']),
        committed_sets=40,solver_records=320,perturbation_records=840,matched_standard_values=checked,
        matched_mean_values=len(means)*3,patients=sorted(patients),
        actual_injections=720,raw_injections_amplified=summary['raw_amplified'],
        structured_injections_clipped_amplified=sum(r['versus_control_clipped']['mae']>r['versus_control_initial']['mae']
             for r in injected if r['perturbation']=='small_vessel_attenuation'),
        artifacts={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in OUT.iterdir()
                   if f.is_file() and f.suffix in ('.csv','.png','.md','.json') and f.name!='verification.json'})
    (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in verification.items() if k!='artifacts'},ensure_ascii=False))


if __name__ == '__main__':
    main()

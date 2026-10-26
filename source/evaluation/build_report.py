"""Build the complete, frozen 50-patient test analysis; no model changes."""
from collections import Counter
import csv
import html
import json
from pathlib import Path

import markdown
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt,font_manager
import numpy as np

from render_cases import render_case,overview

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'report';FIG=OUT/'figures'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def number(x,n=4):return '未定义' if x is None else f'{x:.{n}f}'


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)


def setup():
    FIG.mkdir(parents=True,exist_ok=True)
    p=Path('C:/Windows/Fonts/msyh.ttc')
    if p.exists():
        font_manager.fontManager.addfont(str(p))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(p)).get_name()
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.unicode_minus':False,'figure.facecolor':'white'})


def save(fig,name):
    fig.savefig(FIG/name,dpi=160,bbox_inches='tight',facecolor='white');plt.close(fig)


def plots(report,val):
    cases=report['cases'];ids=[r['case_id'] for r in cases];x=np.arange(50)
    fig,axes=plt.subplots(2,2,figsize=(15,9),constrained_layout=True)
    for ax,(key,label) in zip(axes.flat,[('ilm_opl_ssim','浅层投影SSIM'),('ilm_opl_pcc','浅层投影PCC'),('mae','体积MAE'),('psnr','体积PSNR / dB')]):
        values=[r['metrics'][key] for r in cases]
        ax.plot(x,values,'o-',ms=4,lw=.8,color='#156f79')
        ax.axhline(report['metrics'][key],ls='--',color='#ba7837',label=f'50人均值 {report["metrics"][key]:.4f}')
        ax.set(title=label,xticks=x[::2],xticklabels=[ids[i] for i in x[::2]],xlabel='固定病例顺序')
        ax.tick_params(axis='x',rotation=75,labelsize=7);ax.grid(alpha=.15);ax.legend(fontsize=9)
    fig.suptitle('全部50位测试患者：每例单独展示，未按结果筛选病例',fontsize=15)
    save(fig,'test_patient_metrics.png')
    fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    metrics=[r['metrics'] for r in cases]
    for ax,gkey,pkey,title in [(axes[0,0],'ilm_opl_gt_mean','ilm_opl_pred_mean','浅层投影均值：预测与真实'),
                               (axes[0,1],'ilm_opl_gt_std','ilm_opl_pred_std','浅层投影标准差：预测与真实')]:
        g=np.array([m[gkey] for m in metrics]);p=np.array([m[pkey] for m in metrics]);upper=max(g.max(),p.max())*1.08
        ax.scatter(g,p,s=28,color='#156f79',alpha=.8);ax.plot([0,upper],[0,upper],'--',color='#89959b')
        ax.set(xlim=(0,upper),ylim=(0,upper),xlabel='真实值',ylabel='预测值',title=title)
    high=[m['high_intensity_mae'] for m in metrics];background=[m['background_mae'] for m in metrics]
    axes[1,0].plot(x,high,'o-',ms=3,label='GT>0.1区域',color='#bc793c')
    axes[1,0].plot(x,background,'o-',ms=3,label='其余区域',color='#156f79')
    axes[1,0].set(title='两类强度区域的MAE',xlabel='病例序号',ylabel='MAE');axes[1,0].legend()
    axes[1,1].scatter([m['high_intensity_fraction'] for m in metrics],[m['high_intensity_squared_error_share'] for m in metrics],color='#156f79',label='平方误差占比')
    axes[1,1].scatter([m['high_intensity_fraction'] for m in metrics],[m['high_intensity_absolute_error_share'] for m in metrics],color='#bc793c',marker='x',label='绝对误差占比')
    axes[1,1].set(title='GT高强度区域：体素比例与误差占比',xlabel='GT>0.1体素比例',ylabel='该区域贡献的误差比例',ylim=(0,1));axes[1,1].legend()
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.suptitle('强度与对比度诊断；GT>0.1不是人工血管标签',fontsize=14)
    save(fig,'test_intensity_diagnostics.png')
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,(key,title) in zip(axes.flat,[('mae','体积MAE'),('psnr','体积PSNR / dB'),('ilm_opl_ssim','浅层SSIM'),('ilm_opl_pcc','浅层PCC')]):
        means=[val['metrics'][key],report['metrics'][key]];cis=[val['distribution'][key]['bootstrap95'],report['distribution'][key]['bootstrap95']]
        error=np.array([[m-c[0],c[1]-m] for m,c in zip(means,cis)]).T
        ax.errorbar([0,1],means,yerr=error,fmt='o',ms=7,capsize=5,color='#156f79')
        ax.set(xticks=[0,1],xticklabels=['验证集10人\n用于选best70','测试集50人\n固定模型评价'],xlim=(-.6,1.6),title=title)
        ax.grid(axis='y',alpha=.15)
    fig.suptitle('同一第70轮模型，不同患者群体；误差条为患者bootstrap95%区间，非配对试验',fontsize=12)
    save(fig,'validation_test_metrics.png')


def main():
    report=read(ROOT/'results/metrics.json');val=read(ROOT/'evidence/validation_metrics.json')
    contract=read(ROOT/'verification/frozen_contract.json');pre=read(ROOT/'verification/preflight.json')
    assert report['split']=='test' and report['patients']==len(report['cases'])==50
    assert report['checkpoint_epoch']==70 and pre['state']=='passed' and pre['test_cases_read']==0
    assert [r['case_id'] for r in report['cases']]==contract['test_ids']
    assert report['checkpoint_original_sha256']==val['provenance']['checkpoint_sha256']
    setup();cases=report['cases'];m=report['metrics'];d=report['distribution'];zero=report['zero_output_reference']['metrics']
    for r in cases:
        if not (FIG/f'{r["case_id"]}_comparison.png').exists():render_case(r)
    overview(cases);plots(report,val)
    keys=sorted(m)
    with (OUT/'per_case_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,['case_id','patient_id','disease','device','role','seconds','timing_note']+keys);w.writeheader()
        for r in cases:
            timing='Includes a long pause/wait; not pure GPU compute time' if r['case_id']=='10321' else 'Wall time includes loading, metrics and any waits; not a throughput benchmark'
            w.writerow({**{k:r[k] for k in ['case_id','patient_id','disease','device','role','seconds']},'timing_note':timing,**r['metrics']})
    with (OUT/'metrics_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['metric','mean','sample_sd','median','min','max','valid_cases','ci95_lower','ci95_upper'];w=csv.DictWriter(f,fields);w.writeheader()
        for key in keys:
            if d[key] is None:continue
            z=d[key].copy();ci=z.pop('bootstrap95');w.writerow(dict(metric=key,**z,ci95_lower=ci[0] if ci else None,ci95_upper=ci[1] if ci else None))
    with (OUT/'case_provenance.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['case_id','patient_id','split','checkpoint_epoch','role','device','input_npz_sha256','layer_array_sha256','views_sha256','original_checkpoint_sha256']
        w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(cases)
    selected=[('mae','体积 MAE ↓',6),('mse','体积 MSE ↓',6),('psnr','体积 PSNR ↑ / dB',3),
              ('ilm_opl_ssim','ILM–OPL SSIM ↑',4),('ilm_opl_pcc','ILM–OPL PCC ↑',4),
              ('zmax_ssim','全Z最大投影 SSIM ↑',4),('zmax_pcc','全Z最大投影 PCC ↑',4),
              ('full_mean_ssim','全Z平均投影 SSIM ↑',4),('full_mean_pcc','全Z平均投影 PCC ↑',4),
              ('opl_bm_ssim','OPL–BM SSIM ↑',4),('opl_bm_pcc','OPL–BM PCC ↑',4)]
    metric_table=table(['指标','50人均值 ± 样本标准差','患者bootstrap95%区间','最小–最大'],[
        [label,f'{d[k]["mean"]:.{n}f} ± {d[k]["sample_sd"]:.{n}f}',
         f'[{d[k]["bootstrap95"][0]:.{n}f}, {d[k]["bootstrap95"][1]:.{n}f}]',
         f'{d[k]["min"]:.{n}f}–{d[k]["max"]:.{n}f}'] for k,label,n in selected])
    comparison=table(['指标','验证集10人','测试集50人','测试−验证'],[[label,number(val['metrics'][k],n),number(m[k],n),f'{m[k]-val["metrics"][k]:+.{n}f}'] for k,label,n in selected[:5]])
    subgroup=[]
    for disease in sorted(report['disease_counts']):
        sub=[r['metrics'] for r in cases if r['disease']==disease]
        subgroup.append([disease,len(sub),number(np.mean([r['mae'] for r in sub]),6),number(np.mean([r['ilm_opl_ssim'] for r in sub])),number(np.mean([r['ilm_opl_pcc'] for r in sub]))])
    subgroup_table=table(['病种字段','人数','体积MAE均值','浅层SSIM均值','浅层PCC均值'],subgroup)
    patient_table=table(['病例','病种','体积MAE','PSNR/dB','浅层SSIM','浅层PCC','完整图'],[
        [r['case_id'],r['disease'],number(r['metrics']['mae'],6),number(r['metrics']['psnr'],3),number(r['metrics']['ilm_opl_ssim']),number(r['metrics']['ilm_opl_pcc']),f'[查看](figures/{r["case_id"]}_comparison.png)'] for r in cases])
    highest=max(cases,key=lambda r:r['metrics']['mae']);low=min(cases,key=lambda r:r['metrics']['ilm_opl_ssim']);high=max(cases,key=lambda r:r['metrics']['ilm_opl_ssim'])
    over=[r for r in cases if r['metrics']['ilm_opl_pred_mean']>r['metrics']['ilm_opl_gt_mean']]
    under=[r for r in cases if r['metrics']['ilm_opl_pred_mean']<=r['metrics']['ilm_opl_gt_mean']]
    contrast=m['ilm_opl_pred_std']/m['ilm_opl_gt_std'];bias=m['ilm_opl_pred_mean']-m['ilm_opl_gt_mean']
    high_error_ratio=m['high_intensity_mae']/m['background_mae']
    gt_ranges=[]
    for r in cases:
        with np.load(ROOT/'results/case_views'/f'{r["case_id"]}.npz',allow_pickle=False) as a:gt_ranges.append(float(a['gt_full_mean'].max()))
    zero_table=table(['参照','体积MAE','PSNR/dB','浅层SSIM'],[
        ['固定第70轮模型',number(m['mae'],6),number(m['psnr'],3),number(m['ilm_opl_ssim'])],
        ['全零输出（数值健全性参照）',number(zero['mae'],6),number(zero['psnr'],3),number(zero['ilm_opl_ssim'])]])
    pretable=table(['核对项目','结果'],[
        ['用于核对的病例','原验证病例10302；测试病例使用数=0'],
        ['原最佳权重epoch','70；没有扫描其他轮次'],['GPU窗口批量',pre['window_batch_size']],
        ['与exp003原始指标的最大绝对差',f'{pre["metric_max_absolute_difference"]:.3g}'],
        ['原始投影像素最大绝对差',f'{max(pre["projection_max_differences"].values()):.3g}'],
        ['预设容差','指标/投影均为1e-4；核对通过后才开始测试'],
        ['病例设备分工',', '.join(f'{k}: {v}例' for k,v in report['device_counts'].items())]])
    fixed_examples=[cases[0],cases[24],cases[49]]
    md=f'''# 起点监督3DFM：固定第70轮模型的50例测试集分析

**本报告是50例测试集评价，不是10例验证集报告，也不是旧版FlowOCT/旧3DFM的历史测试报告。** 模型来自exp001的100轮训练，按原10位验证患者的浅层投影SSIM在测试前选定best70。本次不训练、不重新选模、不根据测试结果调整步数、阈值或损失。

测试汇总完成时间：{report['completed_at']}（北京时间）。实验目录按2026-10-25创建日期保留，跨日继续同一固定评价，不另行混入新实验。

50位患者的宏平均体积MAE为 **{m['mae']:.6f}**，PSNR为 **{m['psnr']:.3f} dB**；ILM–OPL最大投影SSIM为 **{m['ilm_opl_ssim']:.4f}**、PCC为 **{m['ilm_opl_pcc']:.4f}**。这些指标描述像素与投影相似度，不等于细血管拓扑或临床准确性。

## 1. 固定设置与数据边界

| 项目 | 本次实际设置 |
| --- | --- |
| 固定划分 | 140训练、10验证、50测试；患者无交叉 |
| 实际测试 | 原清单全部50人，每人1例，未按效果筛选 |
| 模型 | 体素3D U-Net，base16，2,097,521参数 |
| 训练目标 | 流匹配速度MSE；50%精确t=0，50%随机时间；无投影损失 |
| 最佳轮次 | 验证集事先选定第70轮，本次始终不变 |
| 输入 | OCT；OCTA及层标注只用于输出后的评价 |
| 体积 | ZYX=256×304×304；原始单位固定除255 |
| 推理 | FP32，禁用AMP/TF32，Euler8，原256×64×64窗口，50%重叠 |
| 拼接 | 每一步按原Hann权重融合速度；仅最终截断到[0,1] |
| 保存 | 原始二维投影、指定B-scan、逐病例指标和来源；不重复保存完整三维预测 |

原始最佳模型SHA256：`{report['checkpoint_original_sha256']}`。本地执行用无损提取的同一模型权重，提取文件哈希与原始训练文件哈希分别保留。

## 2. 全部50位测试患者的结果

{metric_table}

每位患者等权，PSNR先逐例计算再宏平均；样本标准差用ddof=1。95%区间为固定模型下、按患者重采样10,000次的bootstrap区间，不包含重新训练、随机种子或选模的不确定性。

![全部测试病例指标](figures/test_patient_metrics.png)

## 3. 与10例验证结果怎样对应

{comparison}

这是同一模型在不同患者群体上的描述性比较，不是配对试验。验证集曾用于选择best70，测试集只在冻结设置后评价；群体构成、病例难度和选模都会影响差值，不能仅由两个均值的差断言过拟合或改进原因。

![验证与测试](figures/validation_test_metrics.png)

## 4. 强度、对比度与细节诊断

浅层预测均值的患者宏平均为 **{m['ilm_opl_pred_mean']:.4f}**，真实均值为 **{m['ilm_opl_gt_mean']:.4f}**，平均差为 **{bias:+.4f}**。其中 **{len(over)}例预测均值高于GT、{len(under)}例低于或等于GT**。不能对全部患者统一提亮后宣称结构问题已经解决。

全体积预测均值却为 **{m['volume_pred_mean']:.6f}**，高于GT的 **{m['volume_gt_mean']:.6f}**；预测>0.1的体素比例为 **{m['pred_high_intensity_fraction']:.2%}**，GT为 **{m['high_intensity_fraction']:.2%}**。全体积偏亮与浅层投影偏暗可以同时存在，说明需要分区域检查背景、信号分布和对比度，而不能简单用全图亮度解释血管恢复。阈值占比不等于血管密度或血管召回率。

浅层投影空间标准差的宏平均比值（预测/GT）为 **{contrast:.3f}**。这是对比度诊断，不是毛细血管召回率或连通性得分。

预先固定GT>0.1的高强度区域平均占体素 **{m['high_intensity_fraction']:.2%}**。其MAE为 **{m['high_intensity_mae']:.6f}**，其余区域为 **{m['background_mae']:.6f}**，两者比值为 **{high_error_ratio:.2f}**。该区域贡献绝对误差的患者平均占比为 **{m['high_intensity_absolute_error_share']:.2%}**，平方误差占比为 **{m['high_intensity_squared_error_share']:.2%}**。这些是实际分解，不能仅以背景体素多就断言背景主导了训练MSE或梯度；GT强度阈值也不是人工血管分割。

这里分解的是最终生成体积的重建误差，不能将其直接解释为训练时速度MSE的分布或梯度贡献。

![强度诊断](figures/test_intensity_diagnostics.png)

当前损失仍没有单独检查细血管分叉、断裂或层内投影的准确性。起点监督约束了从纯OCT起步的映射，但不等于增加真实血流信息或细血管专用监督。缺少输入线索时的MSE平均化、下采样/插值和窗口融合，都是需要独立对照才能分清贡献的解释候选；本报告不根据测试图调整这些设置。

对五页全部50例总览以及病例10305、10347、10430的完整图作目视核对：多数病例的大血管走向和中央低信号区可辨认，但细密网状纹理普遍更平滑、更弱，部分主干显得更粗或更亮。10317虽然浅层SSIM最高，仍不能据此认定细血管最准确；10430的全深度平均图明显偏亮，浅层预测均值却低于GT。10454的浅层预测均值高于GT，与多数病例的偏差方向相反。上述都是图像/数值差异描述，不是血管分割或病变诊断结论。

## 5. 为什么不能只看整体误差或平均投影SSIM

{zero_table}

模型相对全零参照的体积MAE变化为 **{m['mae']/zero['mae']-1:+.1%}**，PSNR变化 **{m['psnr']-zero['psnr']:+.3f} dB**。全零只是检查数值口径的参照，不是训练方法。

本次全深度均值投影SSIM为 **{m['full_mean_ssim']:.4f}**，浅层最大投影SSIM为 **{m['ilm_opl_ssim']:.4f}**。平均投影真实图最大值范围仅为 **{min(gt_ranges):.4f}–{max(gt_ranges):.4f}**，所有SSIM仍固定data_range=1、win_size=7。投影内容和动态范围不同，高均值投影SSIM不能替代细结构准确性，也不能直接等同于其他论文不同定义的SSIM。

## 6. 全部病例图像，无效果筛选

五页总览依照固定病例编号排序；每例左GT、右预测，浅层最大投影共享显示窗[0,0.5]。该窗口可能使较亮值饱和，仅影响展示，指标使用原始浮点值。

'''+''.join(f'![测试总览第{i}页](figures/test_overview_{i}.png)\n\n' for i in range(1,6))+f'''
下面固定展示排序第1、25、50位患者，选择规则与模型效果无关；其余病例均可从下表打开完整图。

'''+''.join(f'### 固定示例 {r["case_id"]}\n\n![完整病例](figures/{r["case_id"]}_comparison.png)\n\n' for r in fixed_examples)+f'''
## 7. 病例差异与原始指标

浅层SSIM最低病例为 **{low['case_id']}（{low['metrics']['ilm_opl_ssim']:.4f}）**，最高为 **{high['case_id']}（{high['metrics']['ilm_opl_ssim']:.4f}）**；体积MAE最高为 **{highest['case_id']}（{highest['metrics']['mae']:.6f}）**。这里只描述本次固定模型的分布，没有据此选模型或调参。

{subgroup_table}

病种字段仅用于追溯和描述。小亚组尤其只有1–2人的情况，不支持病种性能排名或临床结论。

{patient_table}

## 8. 推理与来源核对

{pretable}

逐窗口批量组织只改变计算调度，网络、窗口、时间点和累加顺序不变。GPU与CPU后端并非逐位等价，但在原验证病例上与原推理核对后才开始测试。测试期间不以任何测试患者做程序冒烟、性能模式选择、轮次选择或超参数决定。

本次运行跨越本机长时间暂停/等待与次日恢复；原CPU分工完成后，仅把未完成病例重新分配给空闲CPU，已完成结果保留，科学设置未改。逐病例seconds包含等待，不能作为纯GPU计算时间，也不能由本次不连续运行推算公平吞吐。服务器两张3090仍不可初始化，未用于本次生成。

分层投影复用原定义：标注为6层、MATLAB 1基的原640深度边界，Z256体素中心映射后在闭区间取ILM–OPL/OPL–BM最大值；FULL是256层算术均值，投影显示为XY。GT层标注用于预测与GT的相同事后投影，不进入生成模型。

## 9. 文件与解释边界

- [50例逐病例指标](per_case_metrics.csv)、[汇总及置信区间](metrics_summary.csv)、[逐病例来源与哈希](case_provenance.csv)。
- `results/case_views/`保留未拉伸二维数值及指定B-scan；`verification/`保留固定协议、验证核对、设备、来源和完整性记录。
- 原exp001/003代码和权重不覆盖，不制作交付压缩包；本次不增加训练、投影损失或模型修改。
- 这是既有固定测试划分下、当前固定模型的评价。项目历史上已经评价过旧模型的同一测试集，不能把它包装成全项目从未接触过的一次性外部盲测。
- 本次未使用人工细血管分割标签进行该类评价，不能给出有效的细血管Dice、召回率、clDice或临床安全性结论。图像更锐利或高频更多，也不等于新增纹理就是真实血管。
'''
    (OUT/'test_analysis.md').write_text(md,encoding='utf-8')
    body=markdown.markdown(md,extensions=['tables','fenced_code'])
    body=body.replace('<img ','<img loading="lazy" ')
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>起点监督3DFM · 50例测试分析</title><style>
body{margin:0;background:#f3f6f8;color:#203747;font:17px/1.8 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1450px;margin:auto;padding:36px 28px;background:white}h1{font-size:34px;line-height:1.4}h2{margin-top:56px;padding-top:22px;border-top:2px solid #dce6eb}h3{margin-top:34px}p{max-width:1260px}img{max-width:100%;height:auto;border:1px solid #e0e7ec}table{display:block;max-width:100%;overflow:auto;border-collapse:collapse;font-size:15px}td,th{padding:9px 14px;border-bottom:1px solid #dbe4e9;text-align:left;min-width:85px}th{background:#edf3f6}a{color:#0b6c91}code{background:#edf2f5;padding:2px 4px;overflow-wrap:anywhere}li{margin:7px 0}
</style></head><body><main>'''+body+'</main></body></html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    summary=dict(split='test',patients=50,checkpoint_epoch=70,metrics=m,distribution=d,disease_counts=report['disease_counts'],
        device_counts=report['device_counts'],overbright_cases=[r['case_id'] for r in over],underbright_cases=[r['case_id'] for r in under],
        projection_std_ratio=contrast,projection_mean_bias=bias,highest_volume_error_case=highest['case_id'],lowest_ssim_case=low['case_id'],highest_ssim_case=high['case_id'],
        fixed_display_cases=[r['case_id'] for r in fixed_examples],figures=58,no_training_or_tuning=True,visual_review_required=True)
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Built complete 50-case test report, 58 figures, 3 CSV tables')


if __name__=='__main__':main()

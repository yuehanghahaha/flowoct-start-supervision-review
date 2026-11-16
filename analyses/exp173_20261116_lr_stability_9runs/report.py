"""Readable offline report, with original protocols and missing evidence explicit."""
from datetime import datetime, timezone, timedelta
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import ROOT, META, ORDER, GROUPS, read, write, load_history

D=pd.read_csv(ROOT/'data/validation_metrics.csv')
S=pd.read_csv(ROOT/'data/stability_summary.csv')
N=pd.read_csv(ROOT/'data/native_validation_metrics.csv')
L=pd.read_csv(ROOT/'data/learning_rates.csv')
B=pd.read_csv(ROOT/'data/best_final_standard_metrics.csv')
P=pd.read_csv(ROOT/'data/worst_drop_patient_pairs.csv')


def cell(v,digits=4):
    return '未取得' if pd.isna(v) else f'{v:.{digits}f}'


def table(headers,rows):
    return '<div class="table-scroll"><table><thead><tr>'+''.join('<th>'+escape(str(h))+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def md_table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(str(x).replace('|','／') for x in r)+' |\n' for r in rows)


def stat(n,g='volume',window='latter_half',k='ssim'):
    return S[(S.experiment==f'exp{n:03d}')&(S.group==g)&(S.metric==k)&(S.window==window)].iloc[0]


def triples(n,e,g):
    rows=D[(D.experiment==f'exp{n:03d}')&(D.epoch==e)&(D.group==g)]
    if len(rows)!=3:return '未取得该权重留存记录'
    m=dict(zip(rows.metric,rows.value));return f'{m["mae"]:.6f} / {m["psnr"]:.3f} / {m["ssim"]:.6f}'


def main():
    scope=[]
    lrtext={59:'1e-4；E200起线性下降，E300=0',35:'全部200轮固定2e-4',18:'固定2e-4',148:'固定2e-4',
            154:'1e-4；E200起降至5.94e-5',155:'1e-4；E200起降至5.94e-5',163:'固定2e-4',164:'G固定2e-4；D固定1e-4',40:'2e-4；E102起实际低于初值，E200=2e-6'}
    criteria={59:'作者中心裁块三投影SSIM均值；仅E201–300准入',35:'每轮原随机256³裁块MAE最低（G.train）',18:'完整10人浅层MIP SSIM最高',148:'完整10人浅层MIP SSIM最高',154:'完整10人浅层MIP SSIM最高',155:'完整10人浅层MIP SSIM最高',163:'浅层MIP＋三张层均值SSIM等权平均最高',164:'完整10人浅层MIP SSIM最高',40:'原完整验证逐B-scan MAE最低'}
    for n in [59,35,18,164,163,155,154,148,40]:
        m=META[n]
        scope.append([m['label'],f'{m["epochs"]}×{m["updates"]}',f'E{m["best"]}',lrtext[n],criteria[n]])
    judgments={59:'整卷与主投影较稳',18:'整卷明显跳动；投影也有波动',148:'投影较稳，整卷仍有跳动',
               154:'已观测点较稳；记录稀疏',155:'整卷较稳；主投影明显波动',163:'整卷明显跳动；投影不能代表整卷',164:'整卷与主投影均有明显波动'}
    stability=[]
    for n in ORDER:
        a=stat(n);b=stat(n,'ilm_opl_max')
        stability.append([META[n]['label'],f'E{a.start_epoch}–{a.end_epoch} / {a.points}点',
             f'{a.minimum:.4f}–{a.maximum:.4f}',f'{a.mean_abs_step:.4f}',
             f'{b.minimum:.4f}–{b.maximum:.4f}',f'{b.mean_abs_step:.4f}',judgments[n]])
    drop_rows=[]
    for n,g in [(18,'volume'),(148,'volume'),(163,'volume'),(164,'volume'),(155,'ilm_opl_max')]:
        a=stat(n,g);ea,eb=int(a.worst_from_epoch),int(a.worst_to_epoch)
        p=P[(P.experiment==f'exp{n:03d}')&(P.group==g)]
        lr=L[(L.experiment==f'exp{n:03d}')&(L.epoch==eb)].learning_rate_used.iloc[0]
        drop_rows.append([f'exp{n:03d}',GROUPS[g],f'E{ea}→E{eb}',f'{a.worst_from_value:.6f}→{a.worst_to_value:.6f}',
                f'-{a.max_worsening:.6f}',f'{sum(p.ssim_to<p.ssim_from)}/10',f'{sum(p.mae_to>p.mae_from)}/10',f'{lr:.8g}'])
    core=[]
    for n in ORDER:
        for which,e in [('原best',META[n]['best']),('最终轮（补充）',META[n]['epochs'])]:
            core.append([META[n]['label'],which+f' E{e}',triples(n,e,'volume'),triples(n,e,'ilm_opl_max')])
    native35=stat(35,'author_crop','native_latter_half','mae')
    native40=stat(40,'native_bscan2d','native_latter_half','ssim')
    h=load_history(59)
    before=np.array([x['author_validation']['score'] for x in h if 151<=x['epoch']<=199])
    after=np.array([x['author_validation']['score'] for x in h if 251<=x['epoch']<=300])
    beforemad=float(np.abs(np.diff(before)).mean());aftermad=float(np.abs(np.diff(after)).mean())
    hypotheses=[
      '“指标不稳与学习率有关”是合理候选解释。学习率决定一次更新的尺度，PyTorch也提供验证停滞后降低学习率的机制；这不等于每次波动都由学习率过大引起。现有结果没有只改变学习率的配对对照，不能作因果结论。',
      f'四个FM实验全程2e-4，且后半程仍有明显整卷起伏，值得检查学习率。但相同2e-4下，exp148浅层投影相邻变化仅{stat(148,"ilm_opl_max").mean_abs_step:.4f}，exp018为{stat(18,"ilm_opl_max").mean_abs_step:.4f}；采样、条件、损失与BF16等同时不同。相同标量学习率也不代表不同优化器/网络具有相同有效步长。',
      'exp155在E180→E210的主投影SSIM从0.509291降到0.354735，10/10患者同向变差；此时学习率已从1e-4降到8.91e-5。下降中的学习率仍会伴随大幅回落，因此不能把稳定性当作学习率曲线的直接结果。',
      f'XOCT中心裁块指标的相邻轮平均绝对变化从E151–199的{beforemad:.6f}降到E251–300的{aftermad:.6f}，与降学习率同向。后期训练成熟、GAN状态与学习率一起变化；两个时间窗口也不是同一实验时段，仍不能单独归因。',
      'TransPro exp035实际200轮恒定2e-4。其配置/记录里的衰减不能当作已经作用于训练：首次低值0.0001980198是E200结束后留给E201的。exp040确实降低了学习率，但E101–200的原B-scan MAE仍有波动，且未超过E83的原MAE best；它还改变了教师模式、D缩放、验证流程等，不能用035/040差异估计LR效应。',
      'exp164还有已记录的监督/梯度现象：精确t0分支训练中，CDS目标占主要损失数值，t0总梯度在所有20,000次更新都触发clip1。它说明优化受到多项设置共同影响；损失数值比不是分量梯度比，不能据此断定原因。',
      '选模目标也会放大“投影看着最好、整卷却很差”的现象。exp164原best E150的验证3D SSIM仅0.381905，紧接E155为0.812230；E150依然因浅层投影SSIM0.446775入选。这是原规则的真实取舍，与是否存在学习率问题需分开解释。',
    ]
    methodology=[
      '范围以2026年11月16日21:13前后的服务器只读快照冻结。按编号倒序筛选跑满原训练预算的科学实验：exp164、exp163、exp155、exp154、exp148。分析/测试编号不重复计入训练模型；exp156/159/161/162缺完成训练证据，exp168已停止，exp169/170/171/172在快照时未完成，均不凑入五个。八月TransPro没有完整逐轮历史，不能判断其训练稳定性。',
      '统计描述训练过程中固定验证集的跨轮变化。未读取测试患者数组、未新增推理或训练，未用测试结果判断学习率、重选检查点或决定超参数。测试集一次评估中的患者标准差不能代替跨轮稳定性。',
      '全程曲线保留原验证频率，后半程定义为epoch严格大于原预算的一半。相邻变化量为相邻已测点差的绝对值；并同时保留最大恶化、原始范围、去线性趋势残差标准差和符号翻转次数。原始范围可能包含改善趋势，不能单独当作噪声大小。没有套用通用“稳定/不稳”阈值，也没有对自相关轮次做独立样本显著性检验。',
      'FM按共同E105–200、每5轮比较；XOCT与两组消融补充按共同E150/180/210/240比较，后者仅4点、3次变化。每30轮验证的平稳连线不能排除未采样时段的短时波动。学习率前后比较采用连续窗口，保留每个窗口的点数和真实LR；不把绝对LR与逐轮指标相关性当作因果检验。',
      '标准组均使用完整256×304×304视野、10位固定验证患者、未量化float32预测、固定/255 GT和data_range=1。完整3D SSIM在三维计算；逐B-scan为每个Y位置的[Z,X]切片，304张先卷内再患者平均；PSNR也先分别评分再平均。标准SSIM为uniform win7、sample covariance，不能替代TransPro原Gaussian11结果。',
      '主ILM–OPL投影为沿深度的层内最大值，补充对象分别为OPL–BM最大、全深度均值/最大及三种层内均值。XOCT来源的掩膜/层边界沿原实现，FM沿原native640中心映射和闭区间；其层边界并未在本次强行改写。各来源的窗口、融合、数值后端、训练协议不同，跨模型比较是描述性比较，不是单变量LR对照。',
      'TransPro exp035每轮只有作者随机256³裁块、[-1,1]尺度MAE，且G在训练模式；不存在可据此替代的逐轮完整3D或浅层MIP三指标。exp040原记录为GT min–max归一化的逐B-scan评价，原SSIM数值除100显示；其MAE-V/PSNR-V/SSIM-V为血管加权协议，单独保留在native表，不充作普通en-face投影。该区分与TransPro论文§4.2一致。',
      '所有主best均复核原准则：exp018 E170、exp148 E130、exp154/155 E180、exp163 E190、exp164 E150、exp035 E194、exp040 E83。XOCT以当前默认比较偏好展示作者原准入规则best E213；原交付主权重E300另作最终轮补充。XOCT完整验证仅每5轮，无E213同期标准整卷记录，本次不以E210/E215代替。',
      '精度沿历史实际设置保留：exp018与XOCT系列为float32且TF32关闭；exp148为BF16 autocast、float32状态/损失、TF32允许、无scaler；exp163/164为float32、TF32允许、AMP/scaler关闭、benchmark开、deterministic关。近期TransPro035为float32、AMP/scaler关闭、matmul TF32关、cuDNN TF32开；exp040为另一明确调整的后端配置。历史标志以source协议和运行记录为准，没有重新标为统一严格FP32。',
    ]
    nextstep='如果下一步专门验证学习率，优先在新的编号实验里做一个配对对照：保持exp018或其明确选定的现行基线的网络、初始化、数据/裁块/时间序列、损失、精度、预算和原选模准则一致，仅改变预先声明的学习率日程。训练预算保持完整；固定验证集同时观察3D、B-scan及各投影三指标。具体日程应在实施前固定，现有结果不足以选出“最佳学习率”，也不能保证降低后指标一定改善。当前已交付/在跑实验保持原设置，本报告没有启动该对照。'
    audit=read(ROOT/'verification/numerical_audit.json')
    intro='确实存在指标不稳，主要集中在几组3D流匹配的完整体积结果，以及XOCT无CDS组的浅层投影。现有记录支持把学习率列为排查项，尚不能认定它是主因。'
    headers=['实验','后半程 / 点数','3D SSIM范围','3D相邻平均变化','主投影SSIM范围','投影相邻平均变化','观察']
    sections=[]
    sections.append('<p class="eyebrow">实验记录分析 · 2026年11月16日 · exp173</p><h1>指标的波动，出现在哪里？</h1><p class="lead">'+intro+'</p>')
    sections.append('<p class="muted">8个主实验＋1个TransPro补充版本；原始指标、真实学习率、原best与最终轮分开保留。最近五个完整训练：exp164 / exp163 / exp155 / exp154 / exp148。</p>')
    sections.append('<nav><a href="#summary">波动对照</a><a href="#lr">学习率分析</a><a href="#metrics">完整三指标</a><a href="#protocol">口径与证据</a></nav>')
    sections.append('<section id="summary"><h2>后半程波动对照</h2>'+table(headers,stability)+'<p class="muted">相邻平均变化＝相邻已测点SSIM差的绝对值均值。每30轮验证的两组消融只有4个后半程点，不能与每5轮的数值直接排名。</p><img src="figures/01_stability_overview.png" alt="七个实验的完整3D与ILM–OPL投影SSIM轨迹"><img src="figures/05_matched_cadence_volatility.png" alt="统一采样间隔下的波动比较"></section>')
    sections.append('<section><h2>几次回落在多数患者中同时出现</h2>'+table(['实验','对象','轮次','SSIM均值变化','下降量','SSIM下降人数','MAE变差人数','后一轮实际LR'],drop_rows)+'<p>这些均为同10位验证患者的相邻检查点；没有更换病例。波动不只体现在SSIM单一读数，MAE也在多数患者中同步变差。各对象的PSNR变化见逐患者配对表。</p></section>')
    sections.append('<section id="lr"><h2>学习率能解释多少？</h2>'+''.join('<p>'+x+'</p>' for x in hypotheses)+'<p><a href="https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html">PyTorch学习率调整说明</a>仅提供一般机制，不能作为本项目因果证据。</p><img src="figures/02_learning_rates_native_metrics.png" alt="实际学习率与三套原生指标的轨迹"></section>')
    n35='原随机裁块MAE：后半程范围 '+cell(native35.minimum,6)+'–'+cell(native35.maximum,6)+'，相邻平均绝对变化 '+cell(native35.mean_abs_step,6)+'。全程并非没有尖峰：E23→E24的MAE由0.036545升到0.048736，随后回落。后半程的小幅起伏不能反推前期同样稳定，也不能据此宣布其完整投影稳定。'
    n40='补充exp040：原逐B-scan SSIM后半程范围 '+cell(native40.minimum,6)+'–'+cell(native40.maximum,6)+'，相邻平均绝对变化 '+cell(native40.mean_abs_step,6)+'。前半程E87→E88出现明显尖峰：原B-scan SSIM由0.805244降至0.197413，MAE由0.017410升至0.048974；当时LR仍为2e-4。这是原GT min–max、Gaussian11二维口径，不是3D SSIM。后期波动较小也不能仅归因于降LR，因为训练阶段同时改变。'
    sections.append('<section><h2>TransPro需要单独阅读</h2><p>'+n35+'</p><p>'+n40+'</p><p>八月历史版本的逐轮训练/验证/LR记录缺失，本报告不对其稳定性给出判断。</p></section>')
    sections.append('<section id="metrics"><h2>原best与最终轮：完整三指标</h2><p>每格依次为 MAE↓ / PSNR↑ / SSIM↑，所有组使用同一个原best；最终轮只作补充。这里展示验证集，未混入正式测试。</p>'+table(['实验','检查点','完整3D','主ILM–OPL最大投影'],core)+'<img src="figures/03_core_complete_metrics.png" alt="完整3D与主投影的MAE PSNR SSIM曲线">')
    md_supp=[]
    for g in GROUPS:
        if g in ['volume','ilm_opl_max']:continue
        rows=[]
        for n in ORDER:
            rows.append([META[n]['label'],f'E{META[n]["best"]}',triples(n,META[n]['best'],g),
                         f'E{META[n]["epochs"]}',triples(n,META[n]['epochs'],g)])
        sections.append('<details><summary>'+GROUPS[g]+'：MAE / PSNR / SSIM</summary>'+table(['实验','原best','三指标','最终轮','三指标（补充）'],rows)+'</details>')
        md_supp.append('### '+GROUPS[g]+'\n\n'+md_table(['实验','原best','三指标','最终轮','三指标（补充）'],rows))
    sections.append('<details><summary>逐B-scan与补充投影全轨迹图</summary><img src="figures/04_supplementary_complete_metrics.png" alt="逐B-scan与补充投影的完整三指标曲线"></details>')
    native_rows=[]
    for n in [35,40]:
        x=read(ROOT/f'source/exp{n:03d}/native_best_metrics.json');m=x['patient_mean']
        native_rows.append([META[n]['label'],f'E{META[n]["best"]}',f'{m["MAE"]:.6f} / {m["PSNR"]:.3f} / {m["SSIM"]/100:.6f}',
                            '完整304张B-scan先卷内后患者平均；GT min–max，Gaussian11'])
    sections.append('<details open><summary>TransPro原best的原生B-scan三指标（独立协议）</summary>'+table(['实验','原best','MAE / PSNR / SSIM','口径'],native_rows)+'<p>此处复用exp058保留的同10人完整视野记录。exp035这个完整评价MAE与其逐轮随机裁块MAE不是同一对象。固定/255、win7的普通3D和en-face组未取得，不能从原生B-scan数值替代。</p></details></section>')
    sections.append('<section id="protocol"><h2>实验范围、原选模与指标口径</h2>'+table(['实验','完整预算','原best','实际LR','原准则'],scope)+''.join('<p>'+x+'</p>' for x in methodology)+'<p>完整审计通过：'+str(audit['source_sha256_verified'])+'份来源指纹、1,980条完整训练轮记录、236次标准整卷验证、5,532个标准指标值、原9组best复核。每个标准聚合值均重新核对对应10位患者的均值。另保留TransPro逐轮原生记录与XOCT作者中心裁块记录。</p><p>本任务只读服务器，未上传报告、未申请/释放GPU、未改环境、未改变训练设置；未创建压缩交付包。报告及来源目录可直接复制。</p></section>')
    sections.append('<section><h2>若要检验学习率</h2><p>'+nextstep+'</p></section>')
    sections.append('<section><h2>原始数值与可复核文件</h2><ul>'+''.join('<li><a href="../data/'+f+'">'+label+'</a></li>' for f,label in [
        ('validation_metrics.csv','标准验证指标长表：每轮、每组、三指标'),('learning_rates.csv','逐轮实际学习率与记录字段含义'),
        ('stability_summary.csv','完整/后半程/共同间隔的所有统计'),('best_final_standard_metrics.csv','原best、最终轮与缺失记录'),
        ('worst_drop_patient_pairs.csv','最大回落的逐患者配对'),('patient_validation_metrics.csv','全部标准逐患者指标'),
        ('native_validation_metrics.csv','TransPro与XOCT原生逐轮指标'),('training_losses.csv','原定义训练损失；不跨实验比较绝对量级')])+'</ul><p><a href="../verification/numerical_audit.json">数值审计</a> · <a href="../provenance.json">逐文件来源和哈希</a> · <a href="https://arxiv.org/html/2303.06807v2#S4.SS2">TransPro论文§4.2</a></p></section>')
    css='''body{margin:0;background:#f5f6f4;color:#25333b;font:16px/1.75 "Microsoft YaHei",sans-serif}main{max-width:1250px;margin:auto;padding:50px 45px;background:white}h1{font-size:38px;line-height:1.3;font-weight:650;margin:8px 0 22px}h2{font-size:24px;line-height:1.5;margin-top:0}p{max-width:1100px}section{margin-top:48px;padding-top:28px;border-top:1px solid #dce3e5}.eyebrow{font-size:13px;letter-spacing:1px;color:#48788a}.lead{font-size:21px;max-width:1060px}.muted{color:#667780;font-size:14px}a{color:#1f7088;text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:24px;padding:16px 0}img{width:100%;height:auto;display:block;margin:30px 0}table{border-collapse:collapse;width:100%;font-size:13px;line-height:1.65}th,td{padding:12px 10px;border-bottom:1px solid #e0e6e8;text-align:left;vertical-align:top}th{background:#edf3f4;font-weight:600;white-space:nowrap}tbody tr:nth-child(even){background:#fafbfa}.table-scroll{overflow-x:auto}td{min-width:90px}details{margin:16px 0;padding:15px 18px;border:1px solid #d9e2e5;border-radius:4px}summary{cursor:pointer;font-weight:600}@media(max-width:700px){main{padding:28px 16px}h1{font-size:28px}.lead{font-size:18px}}'''
    report=ROOT/'report';report.mkdir(exist_ok=True)
    (report/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>exp173 · 学习率与指标稳定性</title><style>'+css+'</style><main>'+''.join(sections)+'</main></html>',encoding='utf-8')
    md='# 学习率与指标稳定性：8个主实验＋TransPro补充版本\n\n2026年11月16日，exp173，仅分析既有训练与验证。\n\n'+intro+'\n\n'
    md+='## 后半程对照\n\n'+md_table(headers,stability)+'\n![全程轨迹](figures/01_stability_overview.png)\n\n'
    md+='## 明确回落与逐患者核对\n\n'+md_table(['实验','对象','轮次','SSIM变化','下降量','SSIM下降人数','MAE变差人数','实际LR'],drop_rows)+'\n'
    md+='## 学习率能解释多少\n\n'+'\n\n'.join(hypotheses)+'\n\n[PyTorch学习率调整机制](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)。\n\n'+n35+'\n\n'+n40+'\n\n'
    md+='## 完整三指标\n\n每格为 MAE↓ / PSNR↑ / SSIM↑；验证best为主，最终轮为补充。\n\n'+md_table(['实验','检查点','完整3D','主ILM–OPL最大投影'],core)+'\n\n'+'\n\n'.join(md_supp)+'\n\n'
    md+='### TransPro原生逐B-scan协议\n\n'+md_table(['实验','原best','MAE / PSNR / SSIM','口径'],native_rows)+'\n固定/255、win7的完整3D与普通en-face投影三指标未取得，不用原生B-scan替代。八月版本逐轮历史缺失。\n\n'
    md+='## 范围与口径\n\n'+md_table(['实验','完整预算','原best','实际LR','原准则'],scope)+'\n\n'+'\n\n'.join(methodology)+'\n\n[TransPro论文§4.2](https://arxiv.org/html/2303.06807v2#S4.SS2)。\n\n'
    md+='## 下一步\n\n'+nextstep+'\n\n完整长表、统计、原始出处和缺失记录见本实验data/、source/与verification/。\n'
    (report/'report.md').write_text(md,encoding='utf-8')
    write(ROOT/'verification/report_build.json',dict(passed=True,generated=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        report='report/index.html',full_standard_groups=9,all_three_metrics_explicit=True,missing_values_explicit=True,
        references=['https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html','https://arxiv.org/html/2303.06807v2']))
    print('Built offline HTML and detailed Markdown report')


if __name__=='__main__':main()

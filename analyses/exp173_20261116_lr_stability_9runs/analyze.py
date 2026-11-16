"""Analyze retained validation trajectories; never open patient images or checkpoints."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / 'source'
OUT = ROOT / 'data'
OUT.mkdir(exist_ok=True)
META = {
    59: dict(label='XOCT完整 · exp059', epochs=300, updates=140, best=213, every=5, color='#267893'),
    35: dict(label='TransPro · exp035', epochs=200, updates=140, best=194, every=1, color='#8a6599'),
    18: dict(label='exp018 · 起点＋MIP', epochs=200, updates=200, best=170, every=5, color='#323f5d'),
    148: dict(label='exp148 · 无条件纯FM', epochs=200, updates=200, best=130, every=5, color='#b58c2e'),
    154: dict(label='exp154 · XOCT无MSFF', epochs=240, updates=140, best=180, every=30, color='#519a7a'),
    155: dict(label='exp155 · XOCT无CDS', epochs=240, updates=140, best=180, every=30, color='#d18b37'),
    163: dict(label='exp163 · 分层均值L1', epochs=200, updates=200, best=190, every=5, color='#b55a66'),
    164: dict(label='exp164 · 完整CDS', epochs=200, updates=200, best=150, every=5, color='#7759a4'),
    40: dict(label='TransPro · exp040（补充）', epochs=200, updates=140, best=83, every=1, color='#666666'),
}
ORDER = [59, 18, 148, 154, 155, 163, 164]
GROUPS = {
    'volume': '完整3D体积', 'bscan2d': '逐B-scan二维', 'ilm_opl_max': '主ILM–OPL最大投影',
    'opl_bm_max': 'OPL–BM最大投影', 'full_mean': '全深度均值投影', 'full_max': '全深度最大投影',
    'ilm_opl_mean': 'ILM–OPL层内均值', 'opl_bm_mean': 'OPL–BM层内均值', 'ilm_bm_mean': 'ILM–BM层内均值',
}
PREFIXES = {'volume': 'volume', 'bscan2d': 'bscan2d', 'ilm_opl_max': 'projection_ilm_opl',
            'opl_bm_max': 'projection_opl_bm', 'full_mean': 'projection_full',
            'full_max': 'projection_zmax', 'ilm_opl_mean': 'projection_ilm_opl_mean',
            'opl_bm_mean': 'projection_opl_bm_mean', 'ilm_bm_mean': 'projection_ilm_bm_mean'}
PATIENTS = {'10302','10319','10338','10357','10396','10422','10447','10456','10495','10500'}


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p, data):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_history(n):
    p = SRC/f'exp{n:03d}'
    if n in [59, 164]:
        return sorted([read(f) for f in (p/('runs/epochs' if n==59 else 'training_epochs')).glob('*.json')], key=lambda d:d['epoch'])
    if n in [154,155]:
        return read(p/'epochs.json')
    if (p/'history.json').is_file():
        return read(p/'history.json')
    return [json.loads(t) for t in (p/'history.jsonl').read_text(encoding='utf-8-sig').splitlines() if t.strip()]


def standard_groups(n, d):
    if n in [154,155]:
        return d['patient_macro']['standard']
    if n==59:
        mp = {'fixed_volume':'volume', 'fixed_mean':'full_mean', 'fixed_full':'ilm_bm_mean',
              'fixed_ilm_opl':'ilm_opl_mean', 'fixed_opl_bm':'opl_bm_mean',
              'diagnostic_max_ilm_opl':'ilm_opl_max', 'diagnostic_max_opl_bm':'opl_bm_max'}
        return {target:d['patient_macro'][source] for source,target in mp.items()}
    m = d['metrics']
    return {g:{k:m[f'{p}_{k}'] for k in ['mae','psnr','ssim'] if f'{p}_{k}' in m}
            for g,p in PREFIXES.items() if f'{p}_mae' in m}


def standard_cases(n, d):
    if n in [154,155]:
        return {str(x['case_id']):x['standard'] for x in d['cases']}
    if n==59:
        return {str(x['case_id']):standard_groups(n, {'patient_macro':x}) for x in d['cases']}
    return {str(c):standard_groups(n, {'metrics':m}) for c,m in d['per_patient'].items()}


def stats(rows, n, group, metric, window):
    rows = rows.sort_values('epoch')
    e = rows['epoch'].to_numpy(); v=rows['value'].to_numpy(); diff=np.diff(v)
    larger = metric != 'mae'
    bad = -diff if larger else diff
    j=int(np.argmax(bad)) if len(bad) else None
    residual = v - np.polyval(np.polyfit(e, v, 1), e) if len(v)>2 else v-v.mean()
    result=dict(experiment=f'exp{n:03d}', group=group, metric=metric, window=window,
        start_epoch=int(e[0]), end_epoch=int(e[-1]), points=len(v),
        interval_epochs=int(np.median(np.diff(e))) if len(e)>1 else None,
        minimum=float(v.min()), maximum=float(v.max()), span=float(np.ptp(v)),
        mean=float(v.mean()), linear_detrended_sd=float(np.std(residual,ddof=1)) if len(v)>1 else None,
        mean_abs_step=float(np.abs(diff).mean()) if len(diff) else None,
        p95_abs_step=float(np.quantile(np.abs(diff),.95)) if len(diff) else None,
        max_worsening=max(0.,float(bad[j])) if j is not None else None,
        worst_from_epoch=int(e[j]) if j is not None else None,
        worst_to_epoch=int(e[j+1]) if j is not None else None,
        worst_from_value=float(v[j]) if j is not None else None,
        worst_to_value=float(v[j+1]) if j is not None else None,
        first_to_last=float(v[-1]-v[0]), sign_reversals=int(np.sum(diff[1:]*diff[:-1]<0)),
        note='描述统计；不是显著性检验，不定义通用稳定阈值')
    return result


def main():
    provenance=read(ROOT/'provenance.json')
    for f in provenance['files']:
        assert hashlib.sha256((ROOT/f['retained']).read_bytes()).hexdigest()==f['sha256']
    allrows=[]; native=[]; training=[]; lrrows=[]; validations={}; checks=[]; patient_rows=[]
    for n, meta in META.items():
        hist=load_history(n)
        assert [x['epoch'] for x in hist]==list(range(1,meta['epochs']+1))
        for i,d in enumerate(hist):
            epoch=d['epoch']
            if n==35:
                lr=.0002 if i==0 else hist[i-1]['learning_rate_for_next_epoch']
                recorded=d['learning_rate_for_next_epoch']; meaning='记录为下一轮LR；本轮取上一轮记录，E1用初始LR'
            elif n==163:
                lr=d['learning_rates'][0]; recorded=lr; meaning='本轮记录的实际LR'
            else:
                lr=d.get('learning_rate', d.get('generator_lr',.0002));recorded=lr;meaning='本轮实际LR；FM历史缺逐轮LR时由配置和无scheduler代码核对'
            lrrows.append(dict(experiment=f'exp{n:03d}',epoch=epoch,learning_rate_used=lr,
                               learning_rate_recorded=recorded,record_semantics=meaning,
                               discriminator_lr=d.get('discriminator_lr',lr if n in [35,40,59,154,155] else None)))
            loss=d.get('training_mean',d.get('loss_mean',d.get('mean_losses',d.get('losses',{}))))
            for k,v in loss.items():
                if isinstance(v,(float,int)): training.append(dict(experiment=f'exp{n:03d}',epoch=epoch,term=k,value=v))
            for k in ['mean_loss','mean_fm_mse','total_loss','fm_mse']:
                if k in d: training.append(dict(experiment=f'exp{n:03d}',epoch=epoch,term=k,value=d[k]))
            if n==35:
                native.append(dict(experiment='exp035',epoch=epoch,protocol='author_random256crop_train_mode_minus1_plus1',
                                   group='author_crop',metric='mae',value=d['MAE_author_minus1_plus1']))
            elif n==40:
                for k,v in d['validation'].items():
                    native.append(dict(experiment='exp040',epoch=epoch,protocol='TransPro_fullview_GT_minmax_Bscan_gaussian11',
                        group='native_bscan2d' if k in ['MAE','PSNR','SSIM'] else 'native_vessel_diagnostics',
                        metric=k.lower(),value=v/100 if k in ['SSIM','SSIM-V','VDC'] else v))
            elif n==59:
                for k,v in d['author_validation'].items():
                    native.append(dict(experiment='exp059',epoch=epoch,protocol='author_center_crop_reference_projection_ssim',
                                       group='author_crop',metric=k,value=v))
        checks.append(dict(experiment=f'exp{n:03d}',committed_epochs=len(hist),updates=meta['epochs']*meta['updates']))
        if n in [35,40]:continue
        p=SRC/f'exp{n:03d}'/('reports/validation' if n==59 else 'validation')
        files=sorted(p.glob('*/metrics.json' if n==59 else '*.json'))
        assert len(files)==meta['epochs']//meta['every']
        validations[n]={}
        for f in files:
            d=read(f);e=d['epoch'];groups=standard_groups(n,d);cases=standard_cases(n,d)
            assert set(cases)==PATIENTS
            validations[n][e]=dict(groups=groups,cases=cases,source=f.relative_to(ROOT).as_posix())
            for g,metrics in groups.items():
                if g not in GROUPS:continue
                for k in ['mae','psnr','ssim']:
                    value=metrics[k]; assert np.isfinite(value)
                    patient_mean=float(np.mean([cases[c][g][k] for c in PATIENTS]))
                    assert np.isclose(value, patient_mean,rtol=1e-10,atol=1e-10),(n,e,g,k,value,patient_mean)
                    allrows.append(dict(experiment=f'exp{n:03d}',epoch=e,group=g,metric=k,value=value,
                                        patients=10,split='validation',data_range=1.,source=f.relative_to(ROOT).as_posix()))
            for c,cgroups in cases.items():
                for g,m in cgroups.items():
                    if g in GROUPS:
                        for k in ['mae','psnr','ssim']:
                            patient_rows.append(dict(experiment=f'exp{n:03d}',epoch=e,case_id=c,group=g,metric=k,value=m[k]))
    full=pd.DataFrame(allrows); nat=pd.DataFrame(native); lrs=pd.DataFrame(lrrows)
    full.to_csv(OUT/'validation_metrics.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(patient_rows).to_csv(OUT/'patient_validation_metrics.csv',index=False,encoding='utf-8-sig')
    nat.to_csv(OUT/'native_validation_metrics.csv',index=False,encoding='utf-8-sig')
    lrs.to_csv(OUT/'learning_rates.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(training).to_csv(OUT/'training_losses.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for (name,g,k),rows in full.groupby(['experiment','group','metric'],sort=False):
        n=int(name[3:]);summary.append(stats(rows,n,g,k,'full'))
        summary.append(stats(rows[rows.epoch>META[n]['epochs']/2],n,g,k,'latter_half'))
        if n in [59,154,155]:
            shared=rows[rows.epoch.isin([30,60,90,120,150,180,210,240])]
            summary.append(stats(shared,n,g,k,'shared_30_to_240'))
            summary.append(stats(shared[shared.epoch>120],n,g,k,'shared_late_150_to_240'))
    for (name,g,k),rows in nat.groupby(['experiment','group','metric'],sort=False):
        n=int(name[3:]);summary.append(stats(rows,n,g,k,'native_full'))
        summary.append(stats(rows[rows.epoch>META[n]['epochs']/2],n,g,k,'native_latter_half'))
    ss=pd.DataFrame(summary);ss.to_csv(OUT/'stability_summary.csv',index=False,encoding='utf-8-sig')
    selected=[]
    for n,meta in META.items():
        for which,epoch in [('original_best',meta['best']),('final_supplement',meta['epochs'])]:
            for g in GROUPS:
                for k in ['mae','psnr','ssim']:
                    rows=full[(full.experiment==f'exp{n:03d}')&(full.epoch==epoch)&(full.group==g)&(full.metric==k)]
                    value=float(rows.iloc[0].value) if len(rows) else None
                    selected.append(dict(experiment=f'exp{n:03d}',checkpoint=which,epoch=epoch,group=g,metric=k,value=value,
                        availability='retained' if value is not None else '未取得该权重该组留存数值；未补推理'))
    pd.DataFrame(selected).to_csv(OUT/'best_final_standard_metrics.csv',index=False,encoding='utf-8-sig')
    spikes=[]
    for n in ORDER:
        for g in ['volume','ilm_opl_max']:
            row=ss[(ss.experiment==f'exp{n:03d}')&(ss.group==g)&(ss.metric=='ssim')&(ss.window=='latter_half')].iloc[0]
            ea,eb=int(row.worst_from_epoch),int(row.worst_to_epoch)
            a=validations[n][ea];b=validations[n][eb]
            for c in sorted(PATIENTS):
                spikes.append(dict(experiment=f'exp{n:03d}',group=g,from_epoch=ea,to_epoch=eb,case_id=c,
                     ssim_from=a['cases'][c][g]['ssim'],ssim_to=b['cases'][c][g]['ssim'],
                     mae_from=a['cases'][c][g]['mae'],mae_to=b['cases'][c][g]['mae'],
                     psnr_from=a['cases'][c][g]['psnr'],psnr_to=b['cases'][c][g]['psnr']))
    pd.DataFrame(spikes).to_csv(OUT/'worst_drop_patient_pairs.csv',index=False,encoding='utf-8-sig')
    # Check the original selection criterion, without using final/test results to select a weight.
    selection=[]
    for n in ORDER:
        if n==59:
            h=load_history(n);candidates=[(d['epoch'],d['author_validation']['score']) for d in h if d['epoch']>200]
        else:
            candidates=[]
            for e,d in validations[n].items():
                score=np.mean([d['groups'][g]['ssim'] for g in ['ilm_opl_max','ilm_opl_mean','opl_bm_mean','ilm_bm_mean']]) if n==163 else d['groups']['ilm_opl_max']['ssim']
                candidates.append((e,float(score)))
        chosen=max(sorted(candidates),key=lambda p:p[1]);assert chosen[0]==META[n]['best'],(n,chosen)
        selection.append(dict(experiment=f'exp{n:03d}',original_best_epoch=chosen[0],original_selection_value=chosen[1]))
    for n in [35,40]:
        h=load_history(n);key='MAE_author_minus1_plus1' if n==35 else 'MAE'
        chosen=min(h,key=lambda x:x[key] if n==35 else x['validation'][key]);assert chosen['epoch']==META[n]['best']
        selection.append(dict(experiment=f'exp{n:03d}',original_best_epoch=chosen['epoch'],original_selection_value=chosen[key] if n==35 else chosen['validation'][key]))
    assert np.allclose(lrs[lrs.experiment=='exp035'].learning_rate_used,.0002)
    for n in [59,154,155]:
        r=lrs[lrs.experiment==f'exp{n:03d}'];expected=.0001*(1-np.maximum(0,r.epoch.to_numpy()+1-200)/101)
        assert np.allclose(r.learning_rate_used,expected,atol=1e-15,rtol=1e-12)
    write(OUT/'experiment_metadata.json', {f'exp{n:03d}':m for n,m in META.items()})
    write(OUT/'original_checkpoint_selection.json', selection)
    write(ROOT/'verification/numerical_audit.json',dict(passed=True,time=datetime.now(timezone(timedelta(hours=8))).isoformat(),
          source_sha256_verified=len(provenance['files']),epoch_counts=checks,
          validation_records=sum(len(d) for d in validations.values()),standard_metric_values=len(full),
          patient_metric_values=len(patient_rows),fixed_validation_patients=sorted(PATIENTS),
          original_best_selection_verified=len(selection),patient_macro_reaggregation_verified=True,
          actual_learning_rate_semantics_verified=True,new_inference=0,test_data_used=False))
    print(ss[(ss.window=='latter_half')&ss.group.isin(['volume','ilm_opl_max'])&(ss.metric=='ssim')][['experiment','group','minimum','maximum','mean_abs_step','max_worsening','worst_from_epoch','worst_to_epoch']].to_string(index=False))
    print('VERIFIED',len(full),'standard values;',sum(len(d) for d in validations.values()),'full-view validation records')


if __name__=='__main__':
    main()

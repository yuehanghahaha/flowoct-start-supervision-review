"""Diagnose intensity drift from retained validation projections and original logs."""
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

ROOT=Path(__file__).resolve().parent
ORDER=[18,163]
EPOCHS={18:[190,195,200],163:[120,125,130]}
LABEL={18:'exp018 起点＋MIP',163:'exp163 分层均值',164:'exp164 完整CDS',148:'exp148 无条件FM',59:'XOCT exp059'}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    meta=read(ROOT/'provenance.json')
    for f in meta['files']:assert hashlib.sha256((ROOT/f['retained']).read_bytes()).hexdigest()==f['sha256']
    rows=[];pairs=[];diffs=[]
    for n in ORDER:
        gt_identity={}
        for epoch in EPOCHS[n]:
            p=ROOT/f'source/exp{n:03d}/epoch_{epoch:04d}';d=read(p/'metrics.json')
            if n==59:
                cases={str(x['case_id']):dict(volume_mae=x['fixed_volume']['mae'],volume_psnr=x['fixed_volume']['psnr'],
                    volume_ssim=x['fixed_volume']['ssim'],projection_full_mae=x['fixed_mean']['mae'],
                    projection_full_psnr=x['fixed_mean']['psnr'],projection_full_ssim=x['fixed_mean']['ssim'],
                    projection_ilm_opl_mae=x['diagnostic_max_ilm_opl']['mae'],projection_ilm_opl_psnr=x['diagnostic_max_ilm_opl']['psnr'],
                    projection_ilm_opl_ssim=x['diagnostic_max_ilm_opl']['ssim']) for x in d['cases']}
            else:cases=d['per_patient']
            assert len(cases)==10
            for cid,m in cases.items():
                with np.load(p/f'{cid}.npz',allow_pickle=False) as a:
                    pred=a['pred_FULL'].astype(np.float64);gt=a['gt_FULL'].astype(np.float64)
                assert pred.shape==gt.shape==(304,304) and np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1
                if cid in gt_identity:assert np.max(np.abs(gt-gt_identity[cid]))<1e-7
                else:gt_identity[cid]=gt.copy()
                error=pred-gt;mae=float(np.abs(error).mean());bias=float(error.mean());ssim=float(structural_similarity(gt,pred,data_range=1.,win_size=7,use_sample_covariance=True))
                assert np.isclose(mae,m['projection_full_mae'],atol=3e-8,rtol=2e-6),(n,epoch,cid,mae,m['projection_full_mae'])
                assert np.isclose(ssim,m['projection_full_ssim'],atol=1e-5,rtol=2e-5),(n,epoch,cid,ssim,m['projection_full_ssim'])
                assert abs(bias)<=m['volume_mae']+1e-7
                positive=(m['volume_mae']+bias)/(2*m['volume_mae'])
                row=dict(experiment=f'exp{n:03d}',epoch=epoch,case_id=str(cid),gt_volume_mean=float(gt.mean()),
                    pred_volume_mean=float(pred.mean()),volume_signed_bias=bias,
                    positive_error_share_of_volume_MAE=positive,negative_error_share_of_volume_MAE=1-positive,
                    mean_projection_bright_ray_fraction=float((error>0).mean()),
                    **{k:float(m[k]) for k in ['volume_mae','volume_psnr','volume_ssim','projection_full_mae','projection_full_psnr','projection_full_ssim','projection_ilm_opl_mae','projection_ilm_opl_psnr','projection_ilm_opl_ssim']})
                rows.append(row)
        for ea,eb in zip(EPOCHS[n][:-1],EPOCHS[n][1:]):
            for cid in sorted(gt_identity):
                with np.load(ROOT/f'source/exp{n:03d}/epoch_{ea:04d}/{cid}.npz') as a:pa=a['pred_FULL'].astype(np.float64)
                with np.load(ROOT/f'source/exp{n:03d}/epoch_{eb:04d}/{cid}.npz') as a:pb=a['pred_FULL'].astype(np.float64)
                delta=pb-pa
                diffs.append(dict(experiment=f'exp{n:03d}',from_epoch=ea,to_epoch=eb,case_id=cid,
                    volume_mean_shift=float(delta.mean()),mean_projection_ray_increase_fraction=float((delta>0).mean()),
                    mean_projection_shift_sd=float(delta.std()),mean_projection_mean_abs_shift=float(np.abs(delta).mean())))
    data=pd.DataFrame(rows);out=ROOT/'data';out.mkdir(exist_ok=True)
    data.to_csv(out/'validation_intensity_by_patient.csv',index=False,encoding='utf-8-sig')
    summary=data.groupby(['experiment','epoch'],sort=False).mean(numeric_only=True).reset_index()
    summary['positive_bias_patients']=[int(sum((data.experiment==r.experiment)&(data.epoch==r.epoch)&(data.volume_signed_bias>0))) for r in summary.itertuples()]
    summary.to_csv(out/'validation_intensity_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(diffs).to_csv(out/'paired_projection_changes.csv',index=False,encoding='utf-8-sig')
    prior=pd.read_csv(ROOT/'source/prior_analysis/validation_metrics.csv')
    corr=[]
    for n in [18,148,163,164]:
        wide=prior[prior.experiment==f'exp{n:03d}'].pivot(index='epoch',columns=['group','metric'],values='value')
        wide=wide[wide.index>100]
        v=wide[('volume','ssim')];fm=wide[('full_mean','mae')];mip=wide[('ilm_opl_max','ssim')]
        corr.append(dict(experiment=f'exp{n:03d}',points=len(wide),window='E105-E200',
            r_volumeSSIM_vs_fullmeanMAE=float(v.corr(fm)),r_volumeSSIM_vs_mipSSIM=float(v.corr(mip)),
            r_first_difference_volumeSSIM_vs_fullmeanMAE=float(v.diff().corr(fm.diff()))))
    pd.DataFrame(corr).to_csv(out/'descriptive_correlations.csv',index=False,encoding='utf-8-sig')
    selected=pd.concat([prior[(prior.experiment==f'exp{n:03d}')&prior.epoch.isin(es)] for n,es in EPOCHS.items()])
    selected.to_csv(out/'selected_complete_standard_metrics.csv',index=False,encoding='utf-8-sig')
    branches=read(ROOT/'source/exp164_training_branch_summary.json');late=[r for r in branches if r['epoch']>150]
    branchstats=[]
    for name in ['t0','random']:
        group=[r for r in late if r['branch']==name];w=np.array([r['updates'] for r in group])
        branchstats.append(dict(branch=name,epochs=[151,200],updates=int(w.sum()),
             **{k:float(np.average([r[k] for r in group],weights=w)) for k in ['fm_mse','gradient_norm','gradient_clip_fraction']}))
    write(out/'exp164_late_branch_gradients.json',branchstats)
    toy=[]
    for offset in [0.,.005,.01,.015,.02]:
        # For a constant zero GT and constant prediction b, SSIM contrast/structure terms are one.
        actual=float(structural_similarity(np.zeros((9,9,9)),np.full((9,9,9),offset),data_range=1.,win_size=7,use_sample_covariance=True))
        formula=.0001/(offset**2+.0001);assert np.isclose(actual,formula)
        toy.append(dict(offset=offset,SSIM_constant_dark_window=actual,MAE=offset,PSNR=None if offset==0 else float(-20*np.log10(offset))))
    write(out/'ssim_dark_window_illustration.json',toy)
    write(ROOT/'verification/numerical_audit.json',dict(passed=True,sources_verified=len(meta['files']),
        validation_projection_pairs=len(rows),gt_constant_across_epochs_verified=True,
        mean_projection_MAE_and_SSIM_recalculated=True,mean_projection_bias_equals_volume_mean_bias='linearity of full-depth mean; original code definition verified',
        positive_negative_MAE_decomposition='P=(MAE+bias)/2, N=(MAE-bias)/2; signed bias from retained projections',
        scalar_dark_window_formula_verified=True,correlations_descriptive_only=True,new_inference=0,
        test_data_used=False,epochs_are_diagnostic_examples_not_reselected_checkpoints=True,
        time=datetime.now(timezone(timedelta(hours=8))).isoformat()))
    print(summary[['experiment','epoch','volume_signed_bias','volume_mae','volume_psnr','volume_ssim','positive_bias_patients']].to_string(index=False))


if __name__=='__main__':main()

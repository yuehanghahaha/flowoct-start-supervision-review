"""Independently verify public G/H numerical evidence using NumPy only."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
GH = ROOT/'analysis/gh'
MODELS = ('E090','E120','E130','E190')
OBJECTS = ('volume','bscan2d','ILM_OPL','OPL_BM','FULL','ZMAX','ILM_OPL_MEAN','OPL_BM_MEAN','ILM_BM_MEAN')
METRICS = ('mae','psnr','ssim')
SOLVERS = ('Euler1','Euler2','Euler4','Euler8','Euler16','Euler32','Heun4','Heun8')
KINDS = ('offset_+0.005','offset_+0.01','offset_+0.02','offset_-0.02','gaussian','small_vessel_attenuation')


def load(p):return json.loads(p.read_text(encoding='utf-8'))


def rows(name):
    with (GH/name).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def close(a,b):
    if b is None:assert a in ('',None)
    else:assert np.isfinite(float(a)) and abs(float(a)-float(b))<1e-10,(a,b)


def main():
    evidence=load(ROOT/'source/gh_records.json');cfg=load(ROOT/'experiment.json')
    frozen=load(ROOT/'verification/frozen.json');audit=load(GH/'verification.json')
    source_hash=hashlib.sha256((ROOT/'source/gh_records.json').read_bytes()).hexdigest()
    assert source_hash==audit['source_sha256']
    assert hashlib.sha256((ROOT/'verification/frozen.json').read_bytes()).hexdigest()==evidence['frozen_sha256']
    for name,digest in audit['artifacts'].items():assert hashlib.sha256((GH/name).read_bytes()).hexdigest()==digest,name
    cases=set(cfg['validation_cases']);ss=evidence['groups']['solver'];pp=evidence['groups']['perturbation']
    assert evidence['status']['complete'] and len(ss)==320 and len(pp)==840
    assert len(evidence['groups']['sets'])==40 and len(evidence['record_sha256'])==1200
    assert {(r['model'],r['case'],r['solver']+str(r['steps'])) for r in ss}=={(m,c,s) for m in MODELS for c in cases for s in SOLVERS}
    assert {(r['model'],r['case'],r['start_t'],r['perturbation']) for r in pp}=={(m,c,t,k) for m in MODELS for c in cases for t in (.25,.5,.75) for k in ('control',*KINDS)}
    for r in evidence['groups']['sets']:
        assert r['branches']==29 and not r['missing_G6']
        for path,digest in r['files'].items():assert evidence['record_sha256'][path]==digest
    lookup={};groups=defaultdict(list)
    for r in [*ss,*pp]:
        assert r['identity']==dict(model=frozen['models'][r['model']]['tensor_sha256'],input=frozen['inputs'][r['case']]['sha256'])
        assert set(r['standard'])==set(OBJECTS)
        treatment=r['solver']+str(r['steps']) if 'solver' in r else f"G_t{r['start_t']:g}_{r['perturbation']}"
        for obj in OBJECTS:
            lookup[r['model'],r['case'],treatment,obj]=r['standard'][obj]
            groups[r['model'],treatment,obj].append(r['standard'][obj])
    per_patient=rows('per_patient_standard.csv');assert len(per_patient)==len(lookup)==10440
    assert len({tuple(r[k] for k in ('model','case','treatment','object')) for r in per_patient})==10440
    for r in per_patient:
        expected=lookup[r['model'],r['case'],r['treatment'],r['object']]
        for key in METRICS:close(r[key],expected[key])
    means=rows('standard_metrics.csv');assert len(means)==len(groups)==1044
    means_lookup={}
    for r in means:
        key=r['model'],r['treatment'],r['object'];rr=groups[key]
        assert int(r['patients'])==len(rr)==10
        means_lookup[key]={metric:float(np.mean([v[metric] for v in rr])) for metric in METRICS}
        for metric in METRICS:close(r[metric],means_lookup[key][metric])
    for r in rows('recovery_groups.csv'):
        rr=[v for v in pp if v['model']==r['model'] and v['start_t']==float(r['start_t']) and v['perturbation']==r['perturbation']]
        assert len(rr)==int(r['patients'])==10
        close(r['initial_mae_mean'],np.mean([v['versus_control_initial']['mae'] for v in rr]))
        for prefix,field in (('raw_ratio','versus_control_final'),('clipped_ratio','versus_control_clipped')):
            ratios=np.array([v[field]['mae']/v['versus_control_initial']['mae'] for v in rr])
            expected=dict(mean=ratios.mean(),median=np.median(ratios),min=ratios.min(),max=ratios.max(),gt1=(ratios>1).sum(),lt1=(ratios<1).sum())
            for k,v in expected.items():close(r[prefix+'_'+k],v)
        for out,group in (('initial_bias_mean','versus_control_initial'),('final_bias_mean','versus_control_final')):
            close(r[out],np.mean([v[group]['bias'] for v in rr]))
    sl={(r['model'],r['case'],r['solver']+str(r['steps'])):r for r in ss}
    pair_rows=rows('solver_paired_deltas.csv');assert len(pair_rows)==864
    for r in pair_rows:
        m,s,o,k=r['model'],r['solver'],r['object'],r['metric']
        delta=np.array([sl[m,c,s]['standard'][o][k]-sl[m,c,'Euler8']['standard'][o][k] for c in sorted(cases)])
        expected=dict(delta_mean=delta.mean(),delta_median=np.median(delta),delta_min=delta.min(),delta_max=delta.max(),
                      improved=(delta<0 if k=='mae' else delta>0).sum(),worsened=(delta>0 if k=='mae' else delta<0).sum())
        for key,value in expected.items():close(r[key],value)
    hmeans=rows('solver_means.csv');assert len(hmeans)==32
    for r in hmeans:
        rr=[v for v in ss if v['model']==r['model'] and v['solver']+str(v['steps'])==r['solver']]
        for out,field in (('raw_bias','bias'),('negative_ratio','negative_ratio')):close(r[out],np.mean([v['raw'][field] for v in rr]))
        close(r['nfe'],rr[0]['nfe'])
        for obj in OBJECTS:
            for key in METRICS:close(r[obj+'_'+key],means_lookup[r['model'],r['solver'],obj][key])
    runtime=rows('solver_runtime.csv');assert len(runtime)==320
    for r in runtime:
        original=sl[r['model'],r['case'],r['solver']+r['steps']]
        for k in ('nfe','physical_new_calls','seconds'):close(r[k],original[k])
        for k,v in original['raw'].items():close(r['raw_'+k],v)
    pl={(r['model'],r['case'],r['start_t'],r['perturbation']):r for r in pp}
    recovery=rows('perturbation_recovery.csv');assert len(recovery)==840
    for r in recovery:
        original=pl[r['model'],r['case'],float(r['start_t']),r['perturbation']]
        for key in ('nfe','mae_ratio','signed_bias_ratio','seed'):close(r[key],original.get(key))
        for group in ('versus_control_initial','versus_control_final','versus_control_clipped','raw'):
            for k,v in original.get(group,{}).items():close(r[group+'_'+k],v)
    injected=[r for r in pp if r['perturbation']!='control'];assert len(injected)==720
    for r in injected:close(r['mae_ratio'],r['versus_control_final']['mae']/r['versus_control_initial']['mae'])
    assert sum(r['mae_ratio']>1 for r in injected)==720
    structured=[r for r in injected if r['perturbation']=='small_vessel_attenuation']
    assert len(structured)==120 and all(r['versus_control_clipped']['mae']>r['versus_control_initial']['mae'] for r in structured)
    print(json.dumps(dict(passed=True,solver_records=320,perturbation_records=840,committed_sets=40,
        checked_standard_values=31320,checked_patient_mean_values=3132,recovery_groups=72,
        solver_paired_groups=864,actual_injections=720,raw_amplified=720,
        structured_clipped_amplified=120,source_sha256=source_hash,training=0,inference=0)))


if __name__=='__main__':main()

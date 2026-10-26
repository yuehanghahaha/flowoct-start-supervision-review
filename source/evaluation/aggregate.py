"""Aggregate exactly the frozen 50 test patients; never select a model from them."""
from collections import Counter
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def summarize(cases,seed=20261025,repetitions=10000):
    assert len({r['patient_id'] for r in cases})==len(cases)
    keys=sorted(set().union(*(r['metrics'] for r in cases)))
    rng=np.random.default_rng(seed)
    draw=rng.integers(len(cases),size=(repetitions,len(cases)))
    metrics={};distribution={}
    for key in keys:
        values=np.array([r['metrics'].get(key) for r in cases if r['metrics'].get(key) is not None],dtype=float)
        if not len(values):metrics[key]=None;distribution[key]=None;continue
        assert np.isfinite(values).all(),key
        mean=float(values.mean());metrics[key]=mean
        ci=np.percentile(values[draw].mean(1),[2.5,97.5]) if len(values)==len(cases) else None
        distribution[key]=dict(mean=mean,sample_sd=float(values.std(ddof=1)),median=float(np.median(values)),
            min=float(values.min()),max=float(values.max()),valid_cases=len(values),
            bootstrap95=list(map(float,ci)) if ci is not None else None)
    return dict(metrics=metrics,distribution=distribution,patients=len(cases))


def main():
    contract=read(ROOT/'verification/frozen_contract.json');ids=contract['test_ids'];settings=contract['settings']
    records=[];missing=[]
    for cid in ids:
        p=ROOT/'results'/f'{cid}_metrics.json';v=ROOT/'results/case_views'/f'{cid}.npz'
        if not p.exists() or not v.exists():missing.append(cid);continue
        r=read(p)
        assert r['case_id']==cid and r['split']=='test' and r['checkpoint_epoch']==70
        assert r['original_checkpoint_sha256']==settings['original_checkpoint_sha256']
        assert hashlib.sha256(v.read_bytes()).hexdigest()==r['views_sha256']
        assert r['input_npz_sha256']==contract['case_data_sha256'][cid]
        assert r['layer_array_sha256']==contract['case_layer_array_sha256'][cid]
        records.append(r)
    if missing:raise RuntimeError(f'Cannot publish incomplete test report: {len(records)}/50; missing {missing}')
    assert len(records)==50 and len({r['patient_id'] for r in records})==50
    zero=[dict(case_id=r['case_id'],patient_id=r['patient_id'],metrics=r['zero_reference']) for r in records]
    result=summarize(records,settings['bootstrap_seed'],settings['bootstrap_repetitions'])
    result.update(split='test',checkpoint_epoch=70,checkpoint_original_sha256=settings['original_checkpoint_sha256'],
        cases=records,per_patient={r['patient_id']:r['metrics'] for r in records},
        zero_output_reference=summarize(zero,settings['bootstrap_seed'],settings['bootstrap_repetitions']),
        disease_counts=dict(Counter(r['disease'] for r in records)),device_counts=dict(Counter(r['role'] for r in records)),
        summed_case_seconds=float(sum(r['seconds'] for r in records)),
        protocol='Unchanged fixed patient split; equal patient macro mean; all 50 test patients; FP32 Euler8; final clamp; no tuning on test',
        completed_at=datetime.now(timezone(timedelta(hours=8))).isoformat())
    (ROOT/'results/metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    (ROOT/'results/status.json').write_text(json.dumps(dict(state='complete',split='test',completed=50,total=50,checkpoint_epoch=70,
        full_training_started=False,test_used_for_model_selection=False),indent=2),encoding='utf-8')
    print(json.dumps(dict(state='complete',patients=50,metrics=result['metrics']),ensure_ascii=False))


if __name__=='__main__':main()

"""Independent saved-result checks, without rerunning or selecting test models."""
import csv
from html.parser import HTMLParser
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scores(p,g):
    a,b=p.astype(np.float64),g.astype(np.float64);x,y=a-a.mean(),b-b.mean()
    den=np.sqrt((x*x).sum()*(y*y).sum())
    return dict(mae=float(np.abs(a-b).mean()),ssim=float(structural_similarity(a,b,data_range=1,win_size=7)),
                pcc=float((x*y).sum()/den) if den>1e-15 else None)


class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src']:self.links.append(v)


def finite(v):
    if isinstance(v,dict):
        for x in v.values():finite(x)
    elif isinstance(v,list):
        for x in v:finite(x)
    elif isinstance(v,(float,int)):assert math.isfinite(v)


def main():
    contract=read(ROOT/'verification/frozen_contract.json');pre=read(ROOT/'verification/preflight.json')
    report=read(ROOT/'results/metrics.json');status=read(ROOT/'results/status.json');summary=read(ROOT/'report/analysis_summary.json')
    assert status['state']=='complete' and status['completed']==status['total']==50
    assert report['patients']==len(report['cases'])==50 and report['split']=='test' and report['checkpoint_epoch']==70
    assert pre['state']=='passed' and pre['test_cases_read']==0 and pre['validation_case']=='10302'
    assert sha(ROOT/'verification/preflight.json')==contract['validation_preflight_sha256']
    for rel,digest in contract['file_sha256'].items():assert sha(ROOT/rel)==digest,rel
    assert [r['case_id'] for r in report['cases']]==contract['test_ids']
    membership={r['case_id']:r for r in contract['membership']};patients={}
    for r in contract['membership']:
        assert r['patient_id'] not in patients or patients[r['patient_id']]==r['split']
        patients[r['patient_id']]=r['split']
    assert len({r['patient_id'] for r in report['cases']})==50
    plan=read(ROOT/'execution_plan.json')
    assert set(plan['gpu']['case_ids']).isdisjoint(plan['cpu']['case_ids'])
    assert set(plan['gpu']['case_ids'])|set(plan['cpu']['case_ids'])==set(contract['test_ids'])
    deltas=[];checks=[]
    for r in report['cases']:
        cid=r['case_id'];m=r['metrics'];vpath=ROOT/'results/case_views'/f'{cid}.npz'
        assert r['patient_id']==membership[cid]['patient_id'] and membership[cid]['split']=='test'
        assert cid in plan[r['role']]['case_ids']
        assert r['original_checkpoint_sha256']==contract['settings']['original_checkpoint_sha256']
        assert r['input_npz_sha256']==contract['case_data_sha256'][cid]
        assert r['layer_array_sha256']==contract['case_layer_array_sha256'][cid]
        assert sha(vpath)==r['views_sha256']
        with np.load(vpath,allow_pickle=False) as a:
            np.testing.assert_array_equal(a['volume_shape_ZYX'],[256,304,304])
            assert all(a[k].ndim<=2 and np.isfinite(a[k]).all() for k in a.files)
            for name in ['ilm_opl','opl_bm','full_mean','zmax']:
                assert a['pred_'+name].shape==a['gt_'+name].shape==(304,304)
                assert a['pred_'+name].min()>=0 and a['pred_'+name].max()<=1
                calculated=scores(a['pred_'+name],a['gt_'+name])
                for key,value in calculated.items():
                    saved=m[name+'_'+key]
                    assert (value is None)==(saved is None)
                    if value is not None:deltas.append(abs(value-saved));assert abs(value-saved)<1e-10,(cid,name,key)
            assert abs(float(a['pred_ilm_opl'].mean(dtype=np.float64))-m['ilm_opl_pred_mean'])<1e-12
            assert abs(float(a['gt_ilm_opl'].mean(dtype=np.float64))-m['ilm_opl_gt_mean'])<1e-12
        assert abs(-10*math.log10(max(m['mse'],1e-12))-m['psnr'])<1e-10
        f=m['high_intensity_fraction']
        assert abs(f*m['high_intensity_mse']+(1-f)*m['background_mse']-m['mse'])<1e-10
        assert abs(f*m['high_intensity_mae']+(1-f)*m['background_mae']-m['mae'])<1e-7
        assert abs(f*m['high_intensity_mse']/m['mse']-m['high_intensity_squared_error_share'])<1e-10
        assert abs(m['volume_gt_mean']-r['zero_reference']['mae'])<1e-12
        checks.append(dict(case_id=cid,split='test',checkpoint=70,views_sha_verified=True,projection_metrics_recalculated=True))
    for key,value in report['metrics'].items():
        vals=[r['metrics'][key] for r in report['cases'] if r['metrics'].get(key) is not None]
        if vals:assert abs(float(np.mean(vals))-value)<1e-12,key
        else:assert value is None
    finite(report);finite(summary)
    with (ROOT/'report/per_case_metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        csvrows=list(csv.DictReader(f));assert [r['case_id'] for r in csvrows]==contract['test_ids']
        for row,saved in zip(csvrows,report['cases']):assert abs(float(row['ilm_opl_ssim'])-saved['metrics']['ilm_opl_ssim'])<1e-12
    names=[f'{cid}_comparison.png' for cid in contract['test_ids']]+[f'test_overview_{i}.png' for i in range(1,6)]+['test_patient_metrics.png','test_intensity_diagnostics.png','validation_test_metrics.png']
    assert len(names)==58 and {p.name for p in (ROOT/'report/figures').glob('*.png')}==set(names)
    for name in names:
        with Image.open(ROOT/'report/figures'/name) as im:assert min(im.size)>500;im.verify()
    parser=Links();parser.feed((ROOT/'report/index.html').read_text(encoding='utf-8'))
    for link in parser.links:
        if re.match(r'^[A-Za-z]+:',link) or link.startswith('#'):continue
        p=(ROOT/'report'/link.split('#')[0]).resolve()
        assert p.is_relative_to(ROOT) and p.is_file(),link
    assert not list(ROOT.rglob('*.zip')) and not list(ROOT.rglob('*.tar'))
    record=dict(state='passed',test_patients=50,checkpoint_epoch=70,patient_splits_disjoint=True,
        frozen_scientific_files_verified=True,preflight_used_validation_only=True,test_hyperparameter_selection=False,
        projection_metric_max_recalculation_difference=max(deltas),figures=58,links_checked=len(parser.links),
        source_and_view_hashes_verified=True,volume_error_decompositions_verified=True,case_checks=checks,
        no_full_volume_predictions_saved=True,no_delivery_archives=True)
    (ROOT/'verification/report_checks.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in record.items() if k!='case_checks'},ensure_ascii=False))


if __name__=='__main__':main()

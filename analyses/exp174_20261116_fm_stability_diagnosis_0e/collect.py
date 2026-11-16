"""Collect validation-only 2D evidence for an intensity/trajectory diagnosis."""
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent
PARENT=BASE/'exp173_20261116_lr_stability_9runs'
EPOCHS={18:[190,195,200],163:[120,125,130]}
PATIENTS=['10302','10319','10338','10357','10396','10422','10447','10456','10495','10500']


def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    manifest=[]
    def copy(src,dst):
        dst=ROOT/'source'/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        manifest.append(dict(source=str(src.resolve()),retained=dst.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
    for name in ['validation_metrics.csv','patient_validation_metrics.csv','learning_rates.csv','training_losses.csv','original_checkpoint_selection.json']:
        copy(PARENT/'data'/name,'prior_analysis/'+name)
    for n in [18,59,148,154,155,163,164]:
        p=next(BASE.glob(f'exp{n:03d}_*'))
        for rel in ['experiment.json','run.py','method.py','evaluate.py','common.py','fm3d/model.py','fm3d/inference.py']:
            if (p/rel).is_file():copy(p/rel,f'code/exp{n:03d}/{rel}')
        if n==59:
            for name in ['XOCT_model.py','msff3d_generator.py']:
                copy(p/'source/author/models'/name,f'code/exp059/author/models/{name}')
    copy(PARENT/'source/exp164/training_branch_summary.json','exp164_training_branch_summary.json')
    # These are previously retained original validation projections; no 3D array is opened.
    p=Path('D:/data/FSdata/exp018_20261027_3dfm_start_zproj_200e/reports/validation')
    for epoch in EPOCHS[18]:
        folder=next(p.glob(f'epoch_{epoch:04d}_*'))
        copy(folder/'metrics.json',f'exp018/epoch_{epoch:04d}/metrics.json')
        for cid in PATIENTS:
            original=folder/'views'/f'{cid}.npz'
            with np.load(original,allow_pickle=False) as z:
                arrays={k:z[k] for k in ['pred_FULL','gt_FULL']+(['pred_ILM_OPL','gt_ILM_OPL'] if cid=='10302' else [])}
            dst=ROOT/f'source/exp018/epoch_{epoch:04d}/{cid}.npz';dst.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dst,**arrays)
            manifest.append(dict(source=str(original),source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                retained=dst.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
                transformation='retain FULL mean projection pair; retain main MIP pair only for predeclared first validation patient'))
    remote=r'''import base64,datetime,hashlib,io,json,os,pwd
from pathlib import Path
import numpy as np
CFG=__CONFIG__
kind=CFG['kind']
if kind=='hpc':
 h=Path(os.environ['HOME']).resolve();u=pwd.getpwuid(os.getuid()).pw_name
 assert u=='lvyuehang' and str(h)=='/home/lvyuehang'
 os.chdir(h);b=h/'experience';bound=h
else:
 b=Path('/docsystem/niuke/lvyuehang/Flow_oct2octa/experience');bound=b.resolve()
def ck(p):
 assert os.path.commonpath([str(p.resolve()),str(bound)])==str(bound),str(p)
 return p
ck(b);files=[]
assert not any(p.name.startswith('exp174_') for p in b.iterdir())
for number,epochs in CFG['epochs'].items():
 number=int(number);p=ck(next(b.glob('exp%03d_*'%number)))
 for epoch in epochs:
  candidates=list((p/'reports/validation').glob('epoch_%03d'%epoch)) if number==59 else list((p/'reports/validation').glob('epoch_%04d_*'%epoch))
  assert len(candidates)==1,(number,epoch,len(candidates))
  folder=ck(candidates[0]);metric=ck(folder/'metrics.json');raw=metric.read_bytes()
  files.append(dict(source=str(metric),path='exp%03d/epoch_%04d/metrics.json'%(number,epoch),source_sha256=hashlib.sha256(raw).hexdigest(),sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode()))
  for cid in CFG['patients']:
   original=ck(folder/(cid+'_projections.npz') if number==59 else folder/'views'/(cid+'.npz'))
   raw=original.read_bytes()
   with np.load(io.BytesIO(raw),allow_pickle=False) as z:
    if number==59:arrays=dict(pred_FULL=z['prediction_float_mean'],gt_FULL=z['gt_mean']/np.float32(255))
    else:
     keys=['pred_FULL','gt_FULL']+(['pred_ILM_OPL','gt_ILM_OPL'] if cid=='10302' else [])
     arrays={k:z[k] for k in keys}
   buf=io.BytesIO();np.savez_compressed(buf,**arrays);out=buf.getvalue()
   files.append(dict(source=str(original),path='exp%03d/epoch_%04d/%s.npz'%(number,epoch,cid),source_sha256=hashlib.sha256(raw).hexdigest(),sha256=hashlib.sha256(out).hexdigest(),data=base64.b64encode(out).decode(),transformation='retain original full-depth mean pair; XOCT GT mean /255; first FM patient MIP pair retained'))
print(json.dumps(dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),user=pwd.getpwuid(os.getuid()).pw_name,home=os.environ['HOME'],cwd=str(Path.cwd()),bound=str(bound),kind=kind,files=files)))
'''
    write(ROOT/'provenance.json',dict(files=manifest,stage='local_evidence_collected',new_inference=0,test_data_used=False))
    for kind,host,exe,epochs in [
       ('hpc','lvyuehang@10.10.101.35','/home/lvyuehang/.conda/envs/lyh_flowoct/bin/python',{163:EPOCHS[163]})]:
        cfg=dict(kind=kind,epochs=epochs,patients=PATIENTS)
        script=remote.replace('__CONFIG__',repr(cfg))
        r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12',host,exe,'-'],
                          input=script,capture_output=True,text=True,encoding='utf-8',timeout=180,check=True)
        bundle=json.loads(r.stdout)
        for f in bundle.pop('files'):
            dst=ROOT/'source'/f.pop('path');raw=base64.b64decode(f.pop('data'))
            assert hashlib.sha256(raw).hexdigest()==f['sha256'];dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(raw)
            f.update(retained=dst.relative_to(ROOT).as_posix(),host=host);manifest.append(f)
        write(ROOT/f'verification/{kind}_collection.json',bundle)
        print(json.dumps(dict(host=kind,collected_validation_epochs=sum(len(x) for x in epochs.values()))),flush=True)
    write(ROOT/'provenance.json',dict(time=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        files=manifest,patient_arrays_read='existing 2D validation projections only',new_inference=0,
        new_training=0,test_data_used=False,remote_writes=0))
    print(json.dumps(dict(files=len(manifest),validation_projection_pairs=60,new_inference=0)))


if __name__=='__main__':main()

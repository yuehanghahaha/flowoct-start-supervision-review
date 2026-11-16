"""Snapshot retained numerical evidence. No images, model inference, or training."""
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    manifest = []

    def copy(source, destination):
        source = Path(source)
        destination = ROOT / 'source' / destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == digest
        manifest.append(dict(source=str(source.resolve()), retained=destination.relative_to(ROOT).as_posix(),
                             sha256=digest, bytes=destination.stat().st_size))

    def experiment(number):
        return next(BASE.glob(f'exp{number:03d}_*'))

    for n in [18, 35, 40, 59, 148, 154, 155, 163, 164]:
        p = experiment(n)
        for name in ['experiment.json', 'config.json', 'README.md', 'PROTOCOL.md']:
            if (p / name).is_file():
                copy(p / name, f'exp{n:03d}/protocol/{name}')
        for name in ['metrics.py', 'run.py', 'evaluate.py', 'common.py', 'experiment.py', 'training_bridge.py', 'train_author.py', 'schedule.py']:
            if (p / name).is_file():
                copy(p / name, f'exp{n:03d}/code/{name}')

    p = experiment(152) / 'source'
    for n in [18, 148]:
        source = next(p.glob(f'exp{n:03d}_*'))
        for f in (source / 'runs/train').glob('*.json*'):
            if f.name in ['history.json', 'history.jsonl', 'training_complete.json', 'status.json']:
                copy(f, f'exp{n:03d}/{f.name}')
        for f in (source / 'reports/validation').glob('*/metrics.json'):
            copy(f, f'exp{n:03d}/validation/{f.parent.name}.json')

    copy(experiment(44) / 'data/history.json', 'exp035/history.json')
    for rel in ['data/training_complete.json', 'data/best.json', 'data/status.json']:
        copy(experiment(44) / rel, 'exp035/' + Path(rel).name)
    for rel in ['runs/history.json', 'runs/training_complete.json', 'checkpoints/best.json', 'experiment.json']:
        copy(experiment(58) / 'data/a100' / rel, 'exp040/' + Path(rel).name)
    for n, branch, epoch in [(35, 'a800', 194), (40, 'a100', 83)]:
        p = experiment(58) / f'data/{branch}/results/validation/epoch_{epoch:03d}'
        copy(p / 'summary.json', f'exp{n:03d}/native_best_metrics.json')
        for f in (p / 'cases').glob('*.json'):
            copy(f, f'exp{n:03d}/native_best_cases/{f.name}')
    for n in [154, 155]:
        p = experiment(165) / f'data/training/exp{n}'
        for f in p.glob('*.json'):
            copy(f, f'exp{n}/{f.name}')
        for f in (p / 'validation').glob('*.json'):
            copy(f, f'exp{n}/validation/{f.name}')
    copy(experiment(167) / 'verification/parent_training_history.jsonl', 'exp163/history.jsonl')
    copy(experiment(167) / 'runs/train/training_complete.json', 'exp163/training_complete.json')
    copy(experiment(167) / 'verification/parent_validation_best_metrics.json', 'exp163/best_metrics.json')
    p = experiment(166)
    for d in ['training_epochs', 'validation']:
        for f in (p / 'source' / d).glob('*.json'):
            copy(f, f'exp164/{d}/{f.name}')
    for name in ['training_branch_summary.json']:
        copy(p / 'data' / name, f'exp164/{name}')
    copy(experiment(158) / 'verification/log_facts.json', 'exp035/previous_log_audit.json')

    # Existing SSH alias and existing interpreter; only read the established A800 experiment subtree.
    remote = r'''import base64,datetime,hashlib,json,os,pwd
from pathlib import Path
b=Path('/docsystem/niuke/lvyuehang/Flow_oct2octa/experience'); br=b.resolve()
def ck(p):
 assert p.resolve().is_relative_to(br), str(p)
 return p
p=ck(b/'exp059_20261102_xoct_paperlr_300e')
files=[]
targets=list((p/'runs/epochs').glob('epoch_*.json'))
for rel in ['runs/status.json','runs/training_complete.json','experiment.json']:
 targets.append(p/rel)
for d in sorted((p/'reports/validation').iterdir()):
 ck(d)
 for name in ['metrics.json','summary.json']:
  f=ck(d/name)
  if f.is_file(): targets.append(f); break
for f in sorted(set(targets)):
 ck(f)
 if f.is_file():
  raw=f.read_bytes()
  files.append(dict(path=f.relative_to(p).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode()))
inventory=[]
for d in sorted(b.iterdir()):
 if d.name.startswith('exp') and d.name[3:6].isdigit() and int(d.name[3:6])>=148:
  ck(d);item=dict(name=d.name)
  for rel in ['runs/training_complete.json','runs/train/training_complete.json','runs/status.json','runs/train/status.json']:
   f=ck(d/rel)
   if f.is_file():
    v=json.loads(f.read_text());item[rel]={k:v[k] for k in ['state','stage','epoch','epochs','completed_epoch','updates','steps','time'] if k in v}
  inventory.append(item)
assert not any(x['name'].startswith('exp173_') for x in inventory)
print(json.dumps(dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),user=pwd.getpwuid(os.getuid()).pw_name,home=os.environ['HOME'],cwd=str(Path.cwd()),allowed_base=str(br),inventory=inventory,files=files)))
'''
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        'eias-member.eitech.edu.cn',
                        '/docsystem/niuke/lvyuehang/Flow_oct2octa/experience/exp059_20261102_xoct_paperlr_300e/environment/bin/python', '-'],
                       input=remote, capture_output=True, text=True, encoding='utf-8', timeout=120, check=True)
    result = json.loads(r.stdout)
    for item in result.pop('files'):
        destination = ROOT / 'source/exp059' / item['path']
        assert destination.resolve().is_relative_to((ROOT / 'source/exp059').resolve())
        raw = base64.b64decode(item['data'])
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        manifest.append(dict(source=result['allowed_base']+'/exp059_20261102_xoct_paperlr_300e/'+item['path'],
                             host='eias-member.eitech.edu.cn', retained=destination.relative_to(ROOT).as_posix(),
                             sha256=item['sha256'], bytes=len(raw)))
    write(ROOT / 'verification/a800_snapshot.json', result)
    # The HPC host is restricted to this user's persistent home; check every resolved path.
    hpc = r'''import base64,datetime,hashlib,json,os,pwd
from pathlib import Path
u=pwd.getpwuid(os.getuid()).pw_name; h=Path(os.environ['HOME']).resolve()
assert u=='lvyuehang' and str(h)=='/home/lvyuehang'
os.chdir(h)
def ck(p):
 assert os.path.commonpath([str(p.resolve()),str(h)])==str(h), str(p)
 return p
b=ck(h/'experience');inventory=[]
for d in sorted(b.iterdir()):
 if d.name.startswith('exp') and d.name[3:6].isdigit() and int(d.name[3:6])>=148:
  ck(d);item=dict(name=d.name)
  for rel in ['runs/training_complete.json','runs/train/training_complete.json','runs/status.json','runs/train/status.json']:
   f=ck(d/rel)
   if f.is_file():
    v=json.loads(f.read_text());item[rel]={k:v[k] for k in ['state','stage','epoch','epochs','completed_epoch','updates','steps','time'] if k in v}
  inventory.append(item)
assert not any(x['name'].startswith('exp173_') for x in inventory)
p=ck(b/'exp163_20261114_startproj_layermean_a100_200e')
files=[]
for f in sorted((p/'reports/validation').glob('*/metrics.json')):
 ck(f);raw=f.read_bytes()
 files.append(dict(path=f.relative_to(p).as_posix(),sha256=hashlib.sha256(raw).hexdigest(),data=base64.b64encode(raw).decode()))
print(json.dumps(dict(time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),user=u,home=str(h),cwd=str(Path.cwd()),allowed_base=str(b.resolve()),inventory=inventory,files=files)))
'''
    r = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                        'lvyuehang@10.10.101.35', 'python3', '-'],
                       input=hpc, capture_output=True, text=True, encoding='utf-8', timeout=120, check=True)
    result = json.loads(r.stdout)
    for item in result.pop('files'):
        destination = ROOT / 'source/exp163/validation' / (Path(item['path']).parent.name+'.json')
        raw = base64.b64decode(item['data'])
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
        manifest.append(dict(source=result['allowed_base']+'/exp163_20261114_startproj_layermean_a100_200e/'+item['path'],
                             host='10.10.101.35', retained=destination.relative_to(ROOT).as_posix(),
                             sha256=item['sha256'], bytes=len(raw)))
    write(ROOT / 'verification/hpc_snapshot.json', result)
    write(ROOT / 'provenance.json', dict(time=datetime.now(timezone(timedelta(hours=8))).isoformat(),
          files=manifest, analysis_only=True, training_updates=0, new_inference=0,
          patient_arrays_read=0, remote_mutations=0, environment_changes=0, gpu_changes=0))
    print(json.dumps(dict(source_files=len(manifest), total_bytes=sum(x['bytes'] for x in manifest),
                          xoct_epochs=len(list((ROOT/'source/exp059/runs/epochs').glob('*.json'))),
                          xoct_validation_records=len(list((ROOT/'source/exp059/reports/validation').glob('*/metrics.json'))))))


if __name__ == '__main__':
    main()

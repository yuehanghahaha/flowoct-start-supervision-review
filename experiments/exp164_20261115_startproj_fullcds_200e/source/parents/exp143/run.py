"""Frozen per-arm budget training, complete epoch resume, validation and explicit test."""
import argparse
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import signal
import shutil
import sys
import time
import traceback
import numpy as np
import torch
import common as c
import method

ROOT, CKPT, RUN = c.ROOT, c.CKPT, c.RUN


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def restore_optimizers(opts, states, device):
    if len(opts) != len(states): raise ValueError('Optimizer count changed')
    for opt, state in zip(opts, states):
        opt.load_state_dict(state)
        for group in opt.state.values():
            for key, value in group.items():
                if torch.is_tensor(value) and key != 'step': group[key] = value.to(device)


def payload(model, opts, cfg, deployment, epoch, updates, best, ref_best, record, phase):
    return dict(model=model.state_dict(), optimizers=[o.state_dict() for o in opts],
        epoch=epoch, updates=updates, best=best, reference_best=ref_best, epoch_record=record, phase=phase,
        model_tensor_sha256=c.model_digest(model), source_contract=c.source_contract(), config=cfg,
        deployment=deployment, runtime_versions=c.versions(), manifest_sha256=c.sha(deployment['manifest']),
        input_identity=c.read(ROOT/'verification/input_identity.json'), rng_state=c.rng_state())


def check_saved(saved, cfg, deployment):
    if saved['config'] != cfg or saved['source_contract'] != c.source_contract() or saved['deployment'] != deployment:
        raise ValueError('Source/configuration/data binding changed; preserve started experiment')
    if saved['runtime_versions'] != c.versions() or saved['manifest_sha256'] != c.sha(deployment['manifest']):
        raise ValueError('Dependency or manifest identity changed')
    # Verify all already consumed input bytes, including shared read-only caches.
    for path, identity in saved['input_identity'].items():
        if c.sha(path) != identity['sha256']: raise ValueError('Previously consumed input changed: '+path)


def train(cfg, deployment, rows, resume):
    contract=c.source_contract()
    for check in ['preflight.json','full_preflight.json']:
        receipt=c.read(ROOT/'verification'/check)
        if not receipt['passed'] or receipt['source_contract'] != contract:
            raise RuntimeError('Required current check missing/stale: '+check)
    c.configure(cfg); c.seed_all(cfg['seed'])
    device=torch.device(cfg['device'])
    inputs=c.Inputs(deployment,cfg)
    model=method.make_model(cfg).float().to(device)
    opts=method.optimizers(model,cfg)
    CKPT.mkdir(parents=True,exist_ok=True); RUN.mkdir(parents=True,exist_ok=True)
    paths=[CKPT/'initial.pt',CKPT/'latest.pt',CKPT/'trained_pending_validation.pt']
    best=ref_best=None; completed=updates=0; pending=None
    if resume:
        available=[p for p in paths if p.exists()]
        if not available: raise FileNotFoundError('No initial/complete/pending checkpoint')
        # Pending holds a complete trained epoch and can finish interrupted validation.
        recovery=available[-1]
        saved=torch.load(recovery,map_location='cpu',weights_only=False)
        check_saved(saved,cfg,deployment)
        model.load_state_dict(saved['model']); restore_optimizers(opts,saved['optimizers'],device)
        c.restore_rng(saved['rng_state'])
        completed,updates=saved['epoch'],saved['updates']; best,ref_best=saved['best'],saved['reference_best']
        if saved['phase']=='trained_before_validation': pending=saved
        history=RUN/'history.jsonl'
        if history.exists():
            old=[json.loads(s) for s in history.read_text('utf8').splitlines() if s]
            committed=completed-1 if pending else completed
            kept=[v for v in old if v['epoch']<=committed]
            if len(kept)!=len(old):
                shutil.copyfile(history,RUN/f'history_interrupted_{time.time_ns()}.jsonl')
            history.write_text(''.join(json.dumps(v)+'\n' for v in kept),encoding='utf8')
            if not pending and completed and not any(v['epoch']==completed for v in kept):
                with history.open('a',encoding='utf8') as f: f.write(json.dumps(saved['epoch_record'])+'\n')
        if not history.exists() and not pending and completed:
            history.write_text(json.dumps(saved['epoch_record'])+'\n',encoding='utf8')
        for selected,name in [(best,'best.pt'),(ref_best,'best_reference_psnr.pt')]:
            if selected: shutil.copyfile(CKPT/selected['archive'],CKPT/name)
        del saved
    else:
        if any(p.exists() for p in paths): raise FileExistsError('Started run; use resume')
        initial=payload(model,opts,cfg,deployment,0,0,None,None,None,'initial')
        c.save_checkpoint(CKPT/'initial.pt',initial)
        c.write(ROOT/'verification/initialization.json',dict(from_scratch=True,seed=cfg['seed'],
            model_tensor_sha256=initial['model_tensor_sha256'],external_checkpoint=None))
    c.write(ROOT/'verification/run_environment.json',dict(started_at=stamp(),python=sys.version,
        executable=sys.executable,versions=c.versions(),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(device),precision_flags=c.precision_flags(model),
        parameters=sum(p.numel() for p in model.parameters()),config=cfg,source_contract=contract))
    training=[r for r in rows if r['split']=='train']
    start_epoch=completed if pending else completed+1
    for epoch in range(start_epoch,cfg['epochs']+1):
        if pending:
            record=pending['epoch_record']; pending=None
        else:
            tick=time.perf_counter(); c.seed_all(cfg['seed']+epoch); model.train()
            torch.cuda.reset_peak_memory_stats(device)
            indices=np.random.default_rng(cfg['seed']+epoch).integers(len(training),size=cfg['steps_per_epoch'])
            times,_=c.time_schedule(cfg,epoch)
            means={}; origins=[]
            attempt=time.time_ns()
            log=RUN/f'steps_epoch_{epoch:04d}_{attempt}.jsonl'
            with log.open('x',encoding='utf8',buffering=1) as f:
                for step,(index,t) in enumerate(zip(indices,times),1):
                    row=training[int(index)]
                    data=method.batch(inputs,row,cfg,t)
                    values=method.update(model,opts,data,cfg)
                    if not all(math.isfinite(v) for v in values.values()): raise FloatingPointError('Nonfinite step evidence')
                    updates+=1
                    for k,v in values.items(): means[k]=means.get(k,0.)+v/len(indices)
                    geometry={k:v for k,v in data.items() if k not in ('source','target','layers')}
                    f.write(json.dumps(dict(epoch=epoch,step=step,update=updates,case_id=row['case_id'],
                        patient_id=row['patient_id'],geometry=geometry,**values),allow_nan=False)+'\n')
                    if step==1 or step%20==0 or step==len(indices):
                        c.write(RUN/'status.json',dict(state='training',epoch=epoch,completed_epoch=epoch-1,
                            epochs=cfg['epochs'],step=step,updates=updates,pid=os.getpid(),best=best,time=stamp()))
                        print(f'E{epoch}/{cfg["epochs"]} step {step}/{len(indices)} updates={updates} '+json.dumps(values),flush=True)
                    del data
            record=dict(epoch=epoch,updates=updates,training_mean=means,train_seconds=time.perf_counter()-tick,
                training_peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                training_peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                learning_rates=[o.param_groups[0]['lr'] for o in opts],step_evidence=str(log.relative_to(ROOT)))
            c.save_checkpoint(CKPT/'trained_pending_validation.pt',payload(model,opts,cfg,deployment,epoch,updates,
                best,ref_best,record,'trained_before_validation'))
        digest=c.model_digest(model)
        state=c.rng_state(); validation=report=None; improved=ref_improved=False
        try:
            if epoch%cfg['val_every']==0 or epoch==cfg['epochs']:
                c.write(RUN/'status.json',dict(state='validating',epoch=epoch,completed_epoch=epoch-1,
                    updates=updates,pid=os.getpid(),time=stamp(),best=best))
                validation,report=c.evaluate(model,inputs,rows,cfg,'validation',epoch,digest)
                score=validation['metrics'][cfg['selection_metric']]
                ref=validation['metrics']['reference_volume_psnr']
                if score is None or not math.isfinite(score): raise FloatingPointError('Invalid selection score')
                improved=best is None or score>best['value']
                ref_improved=ref is not None and (ref_best is None or ref>ref_best['value'])
                if improved:
                    best=dict(epoch=epoch,metric=cfg['selection_metric'],value=score,report=report,
                        model_tensor_sha256=digest,archive=f'best_epoch_{epoch:04d}_{digest[:12]}.pt')
                if ref_improved:
                    ref_best=dict(epoch=epoch,metric='reference_volume_psnr',value=ref,report=report,
                        model_tensor_sha256=digest,archive=f'reference_best_epoch_{epoch:04d}_{digest[:12]}.pt')
        finally:
            c.restore_rng(state)
        record.update(validation=validation['metrics'] if validation else None,report=report,best=best,time=stamp())
        saved=payload(model,opts,cfg,deployment,epoch,updates,best,ref_best,record,'epoch_complete')
        if improved:
            c.save_checkpoint(CKPT/best['archive'],saved); shutil.copyfile(CKPT/best['archive'],CKPT/'best.pt')
        if ref_improved:
            c.save_checkpoint(CKPT/ref_best['archive'],saved); shutil.copyfile(CKPT/ref_best['archive'],CKPT/'best_reference_psnr.pt')
        c.save_checkpoint(CKPT/'latest.pt',saved)
        if epoch%cfg['save_every']==0 or epoch==cfg['epochs']:
            c.save_checkpoint(CKPT/f'epoch_{epoch:04d}.pt',saved)
        with (RUN/'history.jsonl').open('a',encoding='utf8') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
        c.write(RUN/'status.json',dict(state='complete' if epoch==cfg['epochs'] else 'training',
            epoch=epoch,completed_epoch=epoch,epochs=cfg['epochs'],updates=updates,best=best,pid=os.getpid(),time=stamp()))
        # Pending is temporary only: latest now contains the same full model,
        # optimizers, RNG, input identities, validation and selection state.
        pending_path=CKPT/'trained_pending_validation.pt'
        if pending_path.exists(): pending_path.unlink()
        print(f'Epoch {epoch} committed; best={best}',flush=True)
    c.write(RUN/'training_complete.json',dict(epochs=cfg['epochs'],updates=updates,best=best,
        reference_best=ref_best,time=stamp(),test_started=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['bind','train','resume','status','freeze','test'])
    for name in ['data-root','manifest','labels-root','cache-root']: p.add_argument('--'+name)
    args=p.parse_args(); cfg=c.read(ROOT/'experiment.json')
    if args.action=='status':
        print(json.dumps(c.read(RUN/'status.json') if (RUN/'status.json').exists() else dict(state='not_started'),indent=2));return
    deployment,rows=c.binding(args)
    if args.action=='bind': print(json.dumps(deployment,indent=2)); return
    c.configure(cfg)
    with c.training_lock():
        if args.action in ('train','resume'):
            train(cfg,deployment,rows,args.action=='resume');return
        complete=c.read(RUN/'training_complete.json')
        if complete['epochs']!=cfg['epochs'] or complete['updates']!=cfg['epochs']*cfg['steps_per_epoch']:
            raise ValueError('Budget incomplete')
        path=CKPT/'best.pt'; saved=torch.load(path,map_location='cpu',weights_only=False)
        check_saved(saved,cfg,deployment)
        if saved['model_tensor_sha256']!=complete['best']['model_tensor_sha256']:
            raise ValueError('Main original selection best mismatch')
        freeze=ROOT/'verification/final_test_freeze.json'
        identity=dict(checkpoint='best.pt',checkpoint_sha256=c.sha(path),epoch=saved['epoch'],
            model_tensor_sha256=saved['model_tensor_sha256'],source_contract=c.source_contract(),
            config=cfg,selected_only_with_validation=True)
        if args.action=='freeze':
            if freeze.exists() and c.read(freeze)!=identity: raise ValueError('Existing freeze differs')
            c.write(freeze,identity);return
        if c.read(freeze)!=identity: raise ValueError('Explicit matching freeze required')
        model=method.make_model(cfg).float().to(cfg['device']);model.load_state_dict(saved['model'])
        c.evaluate(model,c.Inputs(deployment,cfg),rows,cfg,'test',saved['epoch'],saved['model_tensor_sha256'])


if __name__=='__main__':
    def interrupted(signum,frame): raise KeyboardInterrupt(f'Signal {signum}; resume last complete/pending epoch')
    signal.signal(signal.SIGTERM,interrupted)
    try: main()
    except BaseException as exc:
        c.write(ROOT/'verification'/f'failure_{time.time_ns()}.json',dict(time=stamp(),error=repr(exc),traceback=traceback.format_exc()))
        raise

"""Train exp018 plus complete XOCT CDS, with immutable full-state recovery."""
import argparse
from datetime import datetime,timezone,timedelta
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

import common as c
from cds import CDS,REGIONS
import method

ROOT,CKPT,RUN = c.ROOT,c.CKPT,c.RUN
LATEST,BEST,PENDING,CANDIDATE = (CKPT/name for name in
    ('latest.pt','best.pt','trained_pending_validation.pt','best_candidate.pt'))


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def binding():
    return c.binding(SimpleNamespace(data_root=None,manifest=None,labels_root=None,cache_root=None))


def status(state, epoch, step, completed, best, **extra):
    c.write(RUN/'status.json',dict(state=state,epoch=epoch,step=step,completed_epoch=completed,
            epochs=200,updates=(epoch-1)*200+step if epoch else 0,best=best,pid=os.getpid(),time=stamp(),**extra))


def tensor_digest(state):
    import hashlib
    h = hashlib.sha256()
    for name,tensor in state.items():
        a = tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode());h.update(a.numpy().tobytes())
    return h.hexdigest()


def payload(model, auxiliary, opts, config, deployment, epoch, best, record, phase):
    return dict(model=model.state_dict(),discriminators=auxiliary.trainable_state(),
                optimizers=[opt.state_dict() for opt in opts],rng_state=c.rng_state(),
                epoch=epoch,updates=epoch*config['steps_per_epoch'],best=best,record=record,phase=phase,
                model_tensor_sha256=c.model_digest(model),discriminator_tensor_sha256=tensor_digest(auxiliary.trainable_state()),
                source_contract=c.source_contract(),config=config,deployment=deployment,runtime_versions=c.versions(),
                manifest_sha256=c.sha(deployment['manifest']),input_identity=c.read(ROOT/'verification/input_identity.json'))


def check_saved(saved, config, deployment):
    if saved['config'] != config or saved['source_contract'] != c.source_contract() or saved['deployment'] != deployment:
        raise ValueError('Started code/configuration/binding changed; use a new experiment')
    if saved['runtime_versions'] != c.versions() or saved['manifest_sha256'] != c.sha(deployment['manifest']):
        raise ValueError('Started runtime or fixed manifest identity changed')
    for name,identity in saved['input_identity'].items():
        if c.sha(name) != identity['sha256']:
            raise ValueError('Previously consumed input changed: '+name)
    if tensor_digest(saved['model']) != saved['model_tensor_sha256']:
        raise ValueError('Saved generator tensors differ from checkpoint identity')
    if tensor_digest(saved['discriminators']) != saved['discriminator_tensor_sha256']:
        raise ValueError('Saved discriminator tensors differ from checkpoint identity')


def load(saved, model, auxiliary, opts):
    model.load_state_dict(saved['model'],strict=True)
    auxiliary.discriminators.load_state_dict(saved['discriminators'],strict=True)
    for optimizer,state in zip(opts,saved['optimizers'],strict=True):
        optimizer.load_state_dict(state)
        for group in optimizer.state.values():
            for key,value in group.items():
                if torch.is_tensor(value) and key != 'step':
                    group[key] = value.to(next(model.parameters()).device)
    c.restore_rng(saved['rng_state'])


def promote(best):
    if best is None:
        return
    for path in (BEST,CANDIDATE):
        if path.is_file():
            value = torch.load(path,map_location='cpu',weights_only=False)
            if value['selection'] == best and tensor_digest(value['model']) == best['model_tensor_sha256']:
                if path == CANDIDATE:
                    CANDIDATE.replace(BEST)
                return
    raise FileNotFoundError('Committed best checkpoint and its pending promotion are missing')


def train(config, deployment, rows):
    c.configure(config)
    contract = c.source_contract()
    for name in ('synthetic.json','gpu_preflight.json','full_preflight.json'):
        receipt = c.read(ROOT/'verification'/name)
        if not receipt['passed'] or receipt['source_contract'] != contract:
            raise RuntimeError('Missing or stale preflight: '+name)
    if c.read(ROOT/'verification/gpu_preflight.json')['runtime_versions'] != c.versions():
        raise RuntimeError('Preflight environment differs from formal runtime')
    c.seed_all(config['seed'])
    device = torch.device(config['device'])
    inputs = c.Inputs(deployment,config)
    model = method.make_model(config).float().to(device)
    initial_digest = c.model_digest(model)
    vgg_path = Path(deployment['torch_home'])/'hub/checkpoints/vgg19-dcbb9e9d.pth'
    if c.sha(vgg_path) != config['cds']['vgg']['weights_sha256']:
        raise ValueError('Frozen VGG supervisor weights changed')
    auxiliary = CDS(config,device)
    opts = method.optimizers(model,auxiliary,config)
    CKPT.mkdir(parents=True,exist_ok=True);RUN.mkdir(parents=True,exist_ok=True)
    completed,best,pending = 0,None,None
    if LATEST.exists():
        saved = torch.load(LATEST,map_location='cpu',weights_only=False)
        check_saved(saved,config,deployment);load(saved,model,auxiliary,opts)
        completed,best = saved['epoch'],saved['best']
        if saved['record'] is not None:
            c.write(RUN/'epochs'/f'epoch_{completed:03d}.json',saved['record'])
        promote(best)
        del saved
    else:
        if BEST.exists() or PENDING.exists():
            raise RuntimeError('Incomplete checkpoint set; refusing an implicit fresh run')
        c.write(ROOT/'verification/initialization.json',dict(from_scratch=True,seed=config['seed'],
                generator_tensor_sha256=initial_digest,generator_parameters=sum(p.numel() for p in model.parameters()),
                discriminator_parameters={name:sum(p.numel() for p in d.parameters()) for name,d in auxiliary.discriminators.items()},
                pretrained_generator_loaded=False,vgg_pretrained_only=True,optimizer_states_empty=all(not o.state for o in opts)))
        c.save_checkpoint(LATEST,payload(model,auxiliary,opts,config,deployment,0,None,None,'epoch_complete'))
    if PENDING.exists():
        candidate = torch.load(PENDING,map_location='cpu',weights_only=False)
        if candidate['epoch'] > completed:
            if candidate['epoch'] != completed+1 or candidate['phase'] != 'trained_before_validation':
                raise ValueError('Pending epoch cannot follow latest committed epoch')
            check_saved(candidate,config,deployment);load(candidate,model,auxiliary,opts)
            pending = candidate
        else:
            PENDING.unlink()
    c.write(ROOT/'verification/run_environment.json',dict(started_at=stamp(),python=sys.version,
            executable=sys.executable,versions=c.versions(),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(device),precision_flags=c.precision_flags(model),
            generator_parameters=sum(p.numel() for p in model.parameters()),source_contract=contract,config=config))
    training = [r for r in rows if r['split']=='train']
    start_epoch = completed+1
    for epoch in range(start_epoch,config['epochs']+1):
        if pending is not None:
            record = pending['record'];pending = None
        else:
            c.seed_all(config['seed']+epoch);model.train()
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            indices = np.random.default_rng(config['seed']+epoch).integers(len(training),size=config['steps_per_epoch'])
            times,_ = c.time_schedule(config,epoch)
            means = {};ledger = ROOT/'logs/epochs'/f'epoch_{epoch:03d}_{time.time_ns()}.jsonl'
            ledger.parent.mkdir(parents=True,exist_ok=True)
            status('training',epoch,0,epoch-1,best)
            with ledger.open('x',encoding='utf-8',buffering=1) as stream:
                for step,(index,sampled_time) in enumerate(zip(indices,times,strict=True),1):
                    row = training[int(index)]
                    data = method.batch(inputs,row,config,float(sampled_time))
                    values = method.update(model,auxiliary,opts,data,config)
                    for key,value in values.items():
                        means[key] = means.get(key,0.)+value/config['steps_per_epoch']
                    stream.write(json.dumps(dict(epoch=epoch,step=step,case_id=row['case_id'],
                                 patient_id=row['patient_id'],origin_ZYX=data['origin_ZYX'],**values),allow_nan=False)+'\n')
                    if step == 1 or step % 20 == 0:
                        status('training',epoch,step,epoch-1,best)
                        print(f'E{epoch}/200 step {step}/200 FM={values["fm_mse"]:.6g} '
                              f'MIP={values["projection_l1"]:.6g} CDS={values["cds_loss"]:.6g} '
                              f't={sampled_time:.6g}',flush=True)
                    del data
            if not math.isclose(means['d_updates']*200,100,abs_tol=1e-7):
                raise RuntimeError('Every epoch must contain exactly100 CDS/D updates and200 FM/G updates')
            record = dict(epoch=epoch,updates=epoch*200,training_mean=means,
                          train_seconds=time.perf_counter()-started,step_evidence=ledger.relative_to(ROOT).as_posix(),
                          peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30,
                          generator_lr=opts[0].param_groups[0]['lr'],discriminator_lr=opts[1].param_groups[0]['lr'])
            if epoch % config['val_every'] == 0 or epoch == config['epochs']:
                c.save_checkpoint(PENDING,payload(model,auxiliary,opts,config,deployment,epoch,best,record,'trained_before_validation'))
        improved = False
        if epoch % config['val_every'] == 0 or epoch == config['epochs']:
            status('validating',epoch,200,epoch-1,best)
            rng = c.rng_state()
            try:
                digest = c.model_digest(model)
                validation,report = c.evaluate(model,inputs,rows,config,'validation',epoch,digest)
            finally:
                c.restore_rng(rng)
            score = validation['metrics'][config['selection_metric']]
            if score is None or not math.isfinite(score):
                raise FloatingPointError('Invalid original validation selection metric')
            record.update(validation=validation['metrics'],report=report)
            if best is None or score > best['value']:
                best = dict(epoch=epoch,value=score,metric=config['selection_metric'],report=report,model_tensor_sha256=digest)
                c.save_checkpoint(CANDIDATE,dict(model=model.state_dict(),selection=best,config=config,
                                                source_contract=contract,runtime_versions=c.versions()))
                improved = True
        record.update(best=best,time=stamp())
        c.save_checkpoint(LATEST,payload(model,auxiliary,opts,config,deployment,epoch,best,record,'epoch_complete'))
        if improved:
            CANDIDATE.replace(BEST)
        c.write(RUN/'epochs'/f'epoch_{epoch:03d}.json',record)
        PENDING.unlink(missing_ok=True)
        status('complete' if epoch == config['epochs'] else 'epoch_committed',epoch,200,epoch,best)
        print(f'Epoch {epoch} committed, best={best}',flush=True)
    c.write(RUN/'training_complete.json',dict(epochs=config['epochs'],updates=config['epochs']*200,
             cds_updates=config['epochs']*100,best=best,time=stamp(),test_started=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','resume','status','freeze','test'])
    args = parser.parse_args()
    if args.action == 'status':
        print(json.dumps(c.read(RUN/'status.json') if (RUN/'status.json').exists() else dict(state='not_started'),indent=2))
        return
    config = c.read(ROOT/'experiment.json')
    deployment,rows = binding()
    with c.training_lock():
        try:
            if args.action in ('freeze','test'):
                complete = c.read(RUN/'training_complete.json')
                if complete['epochs'] != 200 or complete['updates'] != 40000:
                    raise RuntimeError('Formal testing requires the completed200-epoch budget')
                chosen = torch.load(BEST,map_location='cpu',weights_only=False)
                if chosen['selection'] != complete['best'] or chosen['source_contract'] != c.source_contract():
                    raise ValueError('Validation-selected best source/identity mismatch')
                identity = dict(checkpoint='best.pt',checkpoint_sha256=c.sha(BEST),selection=chosen['selection'],
                                source_contract=c.source_contract(),fixed_test_patients=50,test_selects_checkpoint=False)
                freeze = ROOT/'verification/final_test_freeze.json'
                if args.action == 'freeze':
                    if freeze.exists() and c.read(freeze) != identity:
                        raise ValueError('An existing test freeze differs')
                    c.write(freeze,identity)
                    return
                if c.read(freeze) != identity:
                    raise ValueError('Freeze the original selected best before formal testing')
                c.configure(config)
                model = method.make_model(config).float().to(config['device'])
                model.load_state_dict(chosen['model'],strict=True)
                assert c.model_digest(model)==chosen['selection']['model_tensor_sha256']
                test_inputs = c.Inputs(deployment,config)
                test_inputs.labels = ROOT/'test_annotations'
                for name,expected in c.read(ROOT/'source/test_annotation_files.json').items():
                    if c.sha(ROOT/'test_annotations'/name) != expected:
                        raise ValueError('Formal test annotation changed: '+name)
                c.evaluate(model,test_inputs,rows,config,'test',chosen['selection']['epoch'],c.model_digest(model))
                c.write(ROOT/'runs/test_status.json',dict(state='complete',patients=50,best=chosen['selection'],time=stamp()))
                return
            train(config,deployment,rows)
        except BaseException as exc:
            path = RUN/'status.json' if args.action in ('train','resume') else ROOT/'runs/test_status.json'
            state = c.read(path) if path.exists() else {}
            state.update(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=repr(exc),time=stamp())
            c.write(path,state)
            c.write(ROOT/'verification'/f'failure_{time.time_ns()}.json',dict(time=stamp(),traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    def interrupted(signum,frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}; recover latest complete/pending epoch')
    signal.signal(signal.SIGTERM,interrupted)
    main()

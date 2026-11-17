"""Teacher pretraining, full-update atomic recovery, validation and frozen test."""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import math
import numpy as np
import torch
from .runtime import read,write,save,load,digest,seed_for,seed_all,rng_state,restore_rng,amp,paused,Pause,stamp
from .geometry import slab
from .data import patient_schedule,patch_pair
from .flow import noise_for
from .model import VesselTeacher,MicroVesselFlow
from .losses import teacher_loss,patient_objective
from .evaluation import evaluate_split


def optimizer_for(model,lr,weight_decay=.01,betas=(.9,.999),eps=1e-8):
    decay=[];zero=[]
    for p in model.parameters():
        (decay if p.ndim>1 and not getattr(p,'_no_weight_decay',False) else zero).append(p)
    return torch.optim.AdamW([{'params':decay,'weight_decay':weight_decay},{'params':zero,'weight_decay':0.}],lr=lr,betas=betas,eps=eps)


def state(run,phase,**fields):write(Path(run)/'status.json',{'phase':phase,'time':stamp(),**fields})


def teacher_images(dataset,phase,stats,cfg):
    result=[]
    for row in dataset.split(phase):
        _,target,layers=dataset.case(row);mip,_=slab(target,layers,native_depth=cfg['data']['native_layer_depth'])
        maps=dataset.maps(row);labels=maps['capillary'].astype(np.int64)+2*maps['large'].astype(np.int64)
        result.append((row['case_id'],mip[0].numpy()/stats['mip_rms'],labels))
    return result


@torch.inference_mode()
def teacher_validate(model,images,device):
    model.eval();results=[]
    for case,image,labels in images:
        with amp(device):pred=model(torch.from_numpy(image)[None].to(device)).argmax(1)[0].cpu().numpy()
        scores=[]
        for c in (1,2):
            a=pred==c;b=labels==c;scores.append(float((2*(a&b).sum()+1)/(a.sum()+b.sum()+1)))
        results.append({'case_id':case,'capillary_dice':scores[0],'large_dice':scores[1]})
    return {'mean_class_dice':float(np.mean([.5*(r['capillary_dice']+r['large_dice']) for r in results])),'patients':results}


def fit_teacher(dataset,cfg,stats,device,base_signature):
    run=dataset.run;folder=run/'teacher';settings=cfg['teacher']
    if (folder/'complete.json').exists():
        receipt=read(folder/'complete.json')
        if receipt['signature']!=base_signature or digest(folder/'best.pt')!=receipt['sha256']:raise ValueError('Frozen teacher identity changed.')
        model=VesselTeacher(settings['width'],settings['blocks']).to(device)
        model.load_state_dict(load(folder/'best.pt')['model']);return model.eval().requires_grad_(False),receipt
    train=teacher_images(dataset,'train',stats,cfg);validation=teacher_images(dataset,'validation',stats,cfg)
    seed_all(seed_for(cfg['seed'],'teacher_init'));model=VesselTeacher(settings['width'],settings['blocks']).to(device)
    optimizer=optimizer_for(model,settings['lr']);latest=folder/'latest.pt';start=0;best=None;awaiting=False
    if latest.exists():
        payload=load(latest)
        if payload['signature']!=base_signature:raise ValueError('Teacher source/config/input identity changed.')
        model.load_state_dict(payload['model']);optimizer.load_state_dict(payload['optimizer'])
        restore_rng(payload['rng'],device);start=payload['update'];best=payload['best'];awaiting=payload['awaiting_validation']
    def checkpoint(update,pending):
        save(latest,{'signature':base_signature,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                     'rng':rng_state(device),'update':update,'best':best,'awaiting_validation':pending})
    def validate(update):
        nonlocal best
        report=teacher_validate(model,validation,device);write(folder/f'validation_{update:04d}.json',report)
        if best is None or report['mean_class_dice']>best['score']:
            best={'update':update,'score':report['mean_class_dice']};save(folder/'best.pt',{'model':model.state_dict(),'selection':best})
        checkpoint(update,False)
    if awaiting:validate(start)
    for step in range(start+1,settings['updates']+1):
        if paused(run):checkpoint(step-1,False);raise Pause('teacher update boundary')
        model.train();rng=np.random.default_rng(seed_for(cfg['seed'],'teacher_batch',step));x=[];y=[]
        for _ in range(settings['batch_size']):
            _,image,label=train[int(rng.integers(len(train)))];n=settings['crop']
            sy=int(rng.integers(max(1,label.shape[0]-n+1)));sx=int(rng.integers(max(1,label.shape[1]-n+1)))
            x.append(image[:,sy:sy+n,sx:sx+n]);y.append(label[sy:sy+n,sx:sx+n])
        images=torch.as_tensor(np.stack(x),device=device);labels=torch.as_tensor(np.stack(y),device=device)
        optimizer.zero_grad(set_to_none=True)
        lr=settings['lr']*.5*(1+math.cos(math.pi*step/settings['updates']))
        for group in optimizer.param_groups:group['lr']=lr
        with amp(device):loss=teacher_loss(model(images),labels)
        if not torch.isfinite(loss):raise FloatingPointError('Teacher loss is nonfinite.')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        evaluate=step%settings['validation_every']==0 or step==settings['updates']
        if evaluate:
            checkpoint(step,True)
            if paused(run):raise Pause('teacher pending validation saved')
            validate(step)
        if step%20==0:state(run,'teacher',update=step,loss=float(loss))
    if best is None:raise RuntimeError('Teacher must finish validation before freezing.')
    receipt={'signature':base_signature,'selection':best,'sha256':digest(folder/'best.pt'),'updates':settings['updates']}
    write(folder/'complete.json',receipt)
    state(run,'teacher_complete',update=settings['updates'],selection=best)
    model.load_state_dict(load(folder/'best.pt')['model']);return model.eval().requires_grad_(False),receipt


def lr_for(cfg,step):
    s=cfg['train'];warm=s['warmup_updates'];total=s['epochs']*s['patient_groups_per_epoch']
    if step<=warm:return s['lr']*step/max(1,warm)
    progress=(step-warm)/max(1,total-warm)
    return s['min_lr']+.5*(s['lr']-s['min_lr'])*(1+math.cos(math.pi*progress))


def optimizer_update(model,teacher,dataset,group,cfg,stats,device,optimizer,step,aux_override=None):
    optimizer.zero_grad(set_to_none=True);components=[];begin=perf_counter();samplers=[]
    for row,t in group:
        source,target,layers=[v.to(device) for v in dataset.case(row)];maps=dataset.maps(row)
        positions,label=patch_pair(cfg,maps,step,row['case_id']);samplers.append(label)
        noise=noise_for(target.shape,device,seed_for(cfg['seed'],'flow_noise',step,row['case_id']))
        loss,parts=patient_objective(model,teacher,source,target,layers,maps,noise,source.new_tensor([t]),positions,cfg,stats,step,ramp_override=aux_override)
        if not torch.isfinite(loss):raise FloatingPointError('Generator loss is nonfinite.')
        (loss/len(group)).backward();components.append({k:float(v.detach()) for k,v in parts.items()})
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['train']['clip_norm'],error_if_nonfinite=True)
    lr=lr_for(cfg,step)
    for param_group in optimizer.param_groups:param_group['lr']=lr
    optimizer.step()
    return {'update':step,'patients':[r['case_id'] for r,t in group],'times':[t for r,t in group],
            'target_sampler':samplers,'losses':{k:float(np.mean([r[k] for r in components])) for k in components[0]},
            'grad_norm':float(norm),'lr':lr,'seconds':perf_counter()-begin}


@torch.no_grad()
def ema_update(ema,model,decay):
    for a,b in zip(ema.parameters(),model.parameters()):a.lerp_(b,1-decay)


def fit_flow(model,teacher,dataset,cfg,stats,device,signature,teacher_id):
    run=dataset.run;settings=cfg['train'];optimizer=optimizer_for(model,settings['lr'],settings['weight_decay'],tuple(settings['betas']),settings['eps'])
    ema=deepcopy(model).eval().requires_grad_(False);latest=run/'checkpoints/latest.pt';best=None;start=0;epoch_records=[];awaiting=False
    if latest.exists():
        payload=load(latest)
        if payload['signature']!=signature:raise ValueError('Training source/config/teacher/statistics/input changed.')
        model.load_state_dict(payload['model']);ema.load_state_dict(payload['ema']);optimizer.load_state_dict(payload['optimizer'])
        restore_rng(payload['rng'],device);start=payload['update'];best=payload['best'];epoch_records=payload['epoch_records'];awaiting=payload['awaiting_validation']
    def checkpoint(step,pending,records):
        save(latest,{'signature':signature,'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
                     'rng':rng_state(device),'update':step,'epoch_records':records,'best':best,'awaiting_validation':pending})
    def validate(epoch,step):
        nonlocal best
        # Stable identity of the EMA state for resumable patient-level evaluation.
        import hashlib
        h=hashlib.sha256()
        for name,v in ema.state_dict().items():h.update(name.encode());h.update(v.detach().cpu().numpy().tobytes())
        report=evaluate_split(ema,teacher,dataset,'validation',cfg,stats,device,run/f'validation/e{epoch:03d}',h.hexdigest(),teacher_id)
        score=report['selection_score']
        if best is None or score<best['score']:
            best={'epoch':epoch,'update':step,'score':score};save(run/'checkpoints/best.pt',{'signature':signature,'ema':ema.state_dict(),'selection':best})
        checkpoint(step,False,[])
    if awaiting:validate(start//settings['patient_groups_per_epoch'],start);epoch_records=[]
    per_epoch=settings['patient_groups_per_epoch'];total=settings['epochs']*per_epoch
    state(run,'training',epoch=min(start//per_epoch+1,settings['epochs']),update=start,total_updates=total)
    rows=dataset.split('train')
    for epoch in range(start//per_epoch+1,settings['epochs']+1):
        model.train();records=epoch_records if start%per_epoch else [];epoch_records=[]
        groups=list(patient_schedule(cfg,epoch,rows))
        if len(groups)!=per_epoch:raise ValueError('Natural patient-pass epoch must contain 70 two-patient updates.')
        for i,group in enumerate(groups):
            step=(epoch-1)*per_epoch+i+1
            if step<=start:continue
            if paused(run):checkpoint(step-1,False,records);raise Pause('complete optimizer boundary')
            record=optimizer_update(model,teacher,dataset,group,cfg,stats,device,optimizer,step)
            ema_update(ema,model,settings['ema']);records.append(record)
            if step%20==0 or i==0:
                state(run,'training',epoch=epoch,update=step,total_updates=total,loss=record['losses']['total'],seconds=record['seconds'])
                print(f'epoch {epoch:03d} update {step:05d} loss {record["losses"]["total"]:.5f}',flush=True)
            # At the last update, commit the epoch log before exposing its checkpoint.
            if i+1<len(groups) and (step%settings['checkpoint_every']==0 or paused(run)):
                checkpoint(step,False,records)
        write(run/f'train/epoch_{epoch:03d}.json',{'epoch':epoch,'steps':records,'updates':len(records),
              'mean_loss':float(np.mean([r['losses']['total'] for r in records]))})
        pending=epoch%settings['validation_every']==0;checkpoint(epoch*per_epoch,pending,[])
        if paused(run):raise Pause('committed epoch boundary')
        if pending:validate(epoch,epoch*per_epoch)
    receipt={'signature':signature,'epochs':settings['epochs'],'updates':total,'best':best,
             'best_sha256':digest(run/'checkpoints/best.pt'),'teacher_sha256':teacher_id}
    write(run/'training_complete.json',receipt);state(run,'training_complete',**{'epoch':settings['epochs'],'update':total})
    return receipt

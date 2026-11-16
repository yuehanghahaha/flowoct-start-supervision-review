"""Author training calls, with fixed protocol, complete resume, and supplemental reports."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
from common import ROOT,CONFIG,NETS,read,write,now,digest,source_identity,strict_precision
from common import seed_all,options,build_model,checkpoint,restore,environment


def author_validation(model,dataset):
    totals=np.zeros(4,dtype=np.float64);count=0
    with torch.no_grad():
        for data in dataset:
            model.set_input(data);model.test()
            totals += [model.ssim,model.ssim_ilm_opl,model.ssim_opl_bm,model.ssim_mean];count+=1
    assert count==10
    values=totals/count
    assert np.isfinite(values).all(), 'Nonfinite author validation SSIM'
    return dict(zip(['score','ilm_opl_ssim','opl_bm_ssim','mean_ssim'],map(float,values)))


def plot_curves():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[read(p) for p in sorted((ROOT/'runs/epochs').glob('epoch_*.json'))]
    if not records:return
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    x=[r['epoch'] for r in records]
    for k in ['G_L1','G_GAN','G_Perceptual_3d2d_ilm_opl']:
        axes[0].plot(x,[r['loss_mean'][k] for r in records],label=k)
    axes[0].legend(fontsize=7);axes[0].set_title('Training losses (author definitions)')
    axes[1].plot(x,[r['author_validation']['score'] for r in records]);axes[1].set_title('Author central-crop projection SSIM')
    axes[2].plot(x,[r['learning_rate'] for r in records]);axes[2].set_title('Actual learning rate')
    for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
    fig.tight_layout();(ROOT/'reports').mkdir(exist_ok=True)
    fig.savefig(ROOT/'reports/training_curves.png',dpi=160);plt.close(fig)


def train(resume=False):
    from data import create_dataset
    from evaluate import full_validation
    torch.set_num_threads(8);strict_precision()
    identity=source_identity();verified=read(ROOT/'verification/preflight.json')
    assert verified['status']=='passed' and verified['source_identity']==identity, 'Run current preflight first'
    assert read(ROOT/'verification/preparation.json')['status']=='passed'
    membership=read(ROOT/'membership.json');allowed={r['case_id'] for r in membership if r['split']=='train'}
    assert len(allowed)==140
    latest=ROOT/'checkpoints/latest.pt'
    if not resume:
        assert not latest.exists() and not (ROOT/'verification/training_launch.json').exists(), 'Use resume; never overwrite a launched run'
    else:assert latest.exists(), 'Resume requires a complete epoch checkpoint'
    seed_all(CONFIG['seed']);opt=options()
    trainset=create_dataset(opt,phase='train');valset=create_dataset(opt,phase='val')
    assert len(trainset)==140 and len(valset)==10
    model=build_model(opt)
    if resume:
        state=restore(model,latest,identity);start=state['epoch']+1;steps=state['steps'];best=state['best']
        write(ROOT/'verification'/f'resume_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',dict(time=now(),state=state,checkpoint_sha256=digest(latest)))
        previous=state['epoch']
        if previous and previous%CONFIG['full_validation_frequency']==0:
            full_validation(model,opt,previous)
    else:
        start=1;steps=0;best=dict(score=0.,epoch=None)
        checkpoint(model,0,0,best,ROOT/'checkpoints/initial.pt',identity)
        write(ROOT/'verification/training_launch.json',dict(time=now(),pid=os.getpid(),tmux_session='xoct_exp059',
            config=CONFIG,source_identity=identity,environment=environment(),
            formal_training_started=True,test_arrays_opened=False,initial_checkpoint_sha256=digest(ROOT/'checkpoints/initial.pt')))
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal {signum}; resume from last complete epoch')
    signal.signal(signal.SIGTERM,interrupted)
    for epoch in range(start,301):
        begin=time.monotonic();attempt=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        status=dict(stage='training',epoch=epoch,total_epochs=300,completed_epoch=epoch-1,steps=steps,pid=os.getpid(),time=now())
        write(ROOT/'runs/status.json',status)
        # Preserve author scheduler ordering, including its first-epoch warning.
        model.update_learning_rate();lr=model.optimizers[0].param_groups[0]['lr'];losses=[];seen=[]
        path=ROOT/'logs/epochs'/f'epoch_{epoch:03d}_{attempt}.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8',buffering=1) as log:
            for index,data in enumerate(trainset,1):
                step_begin=time.monotonic();cid=data['pids'][0];assert cid in allowed
                model.set_input(data);model.optimize_parameters();torch.cuda.synchronize()
                values=model.get_current_errors();assert all(math.isfinite(v) for v in values.values()), 'Nonfinite loss; stop without skipping updates'
                steps+=1;seen.append(cid);losses.append(values)
                log.write(json.dumps(dict(epoch=epoch,index=index,step=steps,case_id=cid,lr=lr,
                    seconds=time.monotonic()-step_begin,losses=values,time=now()),allow_nan=False)+'\n')
                if index==1 or index%10==0:
                    status.update(step_in_epoch=index,steps=steps,time=now(),last_step_seconds=time.monotonic()-step_begin)
                    write(ROOT/'runs/status.json',status)
                    print(f'Epoch {epoch}/300 step {index}/140 total {steps} lr {lr:.9g} G_L1 {values["G_L1"]:.6f}',flush=True)
                if steps%opt.save_latest_freq==0:model.save('latest')
        assert len(seen)==140 and set(seen)==allowed and len(set(seen))==140
        if epoch%50==0:model.save('latest');model.save(epoch)
        status.update(stage='author_validation',steps=steps,time=now());write(ROOT/'runs/status.json',status)
        val=author_validation(model,valset)
        is_best=epoch>200 and val['score']>best['score']
        if is_best:best=dict(score=val['score'],epoch=epoch);model.save('best')
        summary=dict(epoch=epoch,steps=steps,learning_rate=lr,author_validation=val,best=best,
            loss_mean={k:float(np.mean([v[k] for v in losses])) for k in losses[0]},time=now(),
            train_and_crop_validation_seconds=time.monotonic()-begin)
        # Commit complete author epoch before the independent, repeatable full-volume supplement.
        write(ROOT/'runs/epochs'/f'epoch_{epoch:03d}.json',summary)
        checkpoint(model,epoch,steps,best,latest,identity)
        if is_best:checkpoint(model,epoch,steps,best,ROOT/'checkpoints/best.pt',identity)
        if epoch%50==0:checkpoint(model,epoch,steps,best,ROOT/'checkpoints'/f'epoch_{epoch:03d}.pt',identity)
        if epoch%CONFIG['full_validation_frequency']==0 or epoch==300:
            status.update(stage='full_validation',completed_epoch=epoch,time=now());write(ROOT/'runs/status.json',status)
            report=full_validation(model,opt,epoch)
            summary['full_validation_report']=str((ROOT/'reports/validation'/f'epoch_{epoch:03d}/metrics.json').relative_to(ROOT))
            summary['full_volume_seconds']=time.monotonic()-begin-summary['train_and_crop_validation_seconds']
            write(ROOT/'runs/epochs'/f'epoch_{epoch:03d}.json',summary)
        plot_curves()
        write(ROOT/'runs/status.json',dict(stage='epoch_complete',epoch=epoch,completed_epoch=epoch,total_epochs=300,steps=steps,best=best,time=now(),pid=os.getpid()))
    write(ROOT/'runs/training_complete.json',dict(time=now(),epochs=300,steps=steps,primary='checkpoints/epoch_300.pt',
        primary_sha256=digest(ROOT/'checkpoints/epoch_300.pt'),supplementary_best=best,test_started=False))
    write(ROOT/'runs/status.json',dict(stage='training_complete',completed_epoch=300,total_epochs=300,steps=steps,best=best,time=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['train','resume','status','validate']);p.add_argument('--epoch',type=int)
    args=p.parse_args()
    if args.action=='status':print(json.dumps(read(ROOT/'runs/status.json') if (ROOT/'runs/status.json').exists() else {'stage':'not_started'},indent=2))
    elif args.action=='validate':
        from evaluate import full_validation
        assert args.epoch is not None
        seed_all(CONFIG['seed']);opt=options();model=build_model(opt)
        state=restore(model,ROOT/'checkpoints'/f'epoch_{args.epoch:03d}.pt',source_identity())
        full_validation(model,opt,state['epoch'])
    else:
        try:train(args.action=='resume')
        except BaseException as exc:
            write(ROOT/'runs/failure.json',dict(time=now(),error=repr(exc),traceback=traceback.format_exc(),resume_policy='last complete epoch; preserve partial-attempt logs'))
            status=read(ROOT/'runs/status.json') if (ROOT/'runs/status.json').exists() else {}
            status.update(stage='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',time=now(),error=repr(exc))
            write(ROOT/'runs/status.json',status)
            raise

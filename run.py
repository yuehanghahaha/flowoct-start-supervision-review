"""Exp181 entry: exp176 plus relative-depth sub-slab supervision."""
from pathlib import Path
import argparse
import csv
import importlib.metadata
import json
import sys
from time import perf_counter
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parent
ACTIVE_RUN=None
OWNS_OPERATION=False
sys.path.insert(0,str(ROOT/'src'))
from microflow.runtime import (read,write,load,digest,source_signature,configure,seed_all,
                              operation_lock,Pause,stamp)
from microflow.data import Dataset,statistics,patient_schedule
from microflow.model import MicroVesselFlow
from microflow.engine import fit_teacher,fit_flow,state,optimizer_for,optimizer_update
from microflow.evaluation import evaluate_split
from microflow.geometry import slab


def audit(dataset,cfg):
    for phase in ('train','validation'):
        for row in dataset.split(phase):
            _,target,layers=dataset.case(row);maps=dataset.maps(row)
            image=slab(target,layers,native_depth=cfg['data']['native_layer_depth'])[0][0,0].numpy()
            rgb=np.repeat(np.uint8(np.clip(image/.5,0,1)*255)[...,None],3,axis=-1)
            for key,color in [('capillary',[0,255,0]),('large',[255,0,0]),('faz',[0,64,255])]:
                rgb[maps[key]]=(rgb[maps[key]]*.5+np.array(color)*.5).astype(np.uint8)
            out=dataset.run/'annotation_audit'/phase/(row['case_id']+'.png');out.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(rgb).save(out)
    write(dataset.run/'annotation_audit/status.json',{'automatic_geometry_checks':'passed',
          'visual_alignment':'requires_review_before_formal_start','test_images_read':False,
          'orientation_is_declared_not_inferred':True,'projection':'ILM_OPL_MAX'})


def gpu_check(model,teacher,dataset,cfg,stats,device,signature):
    if device.type!='cuda':raise RuntimeError('Full-size acceptance must use the allocated CUDA GPU.')
    optimizer=optimizer_for(model,cfg['train']['lr']);rows=dataset.split('train')[:2]
    group=[(rows[0],0.),(rows[1],.75)];records=[];torch.cuda.reset_peak_memory_stats(device)
    observed={}
    hook=model.local_output.register_forward_hook(lambda module,args,out:observed.update(output_dtype=str(out.dtype)))
    try:
        model.train()
        for i in range(25):
            if (dataset.run/'pause.request').exists():raise Pause('GPU check boundary')
            record=optimizer_update(model,teacher,dataset,group,cfg,stats,device,optimizer,i+1,aux_override=1.);records.append(record)
            state(dataset.run,'gpu_check',diagnostic_updates=i+1,formal_updates=0)
        report={'state':'training_passed_inference_pending','signature':signature,'time':stamp(),'formal_updates':0,
          'precision':cfg['precision'],'packages':read(dataset.run/'environment.json')['packages'],
          'actual_network_output_dtype':observed.get('output_dtype'),'actual_parameter_dtypes':sorted({str(p.dtype) for p in model.parameters()}),
          'engineering_auxiliary_factor':1.0,
          'training_case_ids':[r['case_id'] for r in rows],'shape':cfg['data']['shape'],
          'patch_input':[c+2*h for c,h in zip(cfg['patch']['core'],cfg['patch']['halo'])],
          'warmup':5,'timed':20,'median_seconds':float(np.median([r['seconds'] for r in records[5:]])),
          'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
          'peak_reserved_bytes':torch.cuda.max_memory_reserved(device),'records':records}
        write(dataset.run/'verification/full_size.json',report)
        total_memory=torch.cuda.get_device_properties(device).total_memory
        if report['peak_reserved_bytes']>total_memory-8*1024**3:
            raise RuntimeError('Measured setup has less than 8GiB headroom; review hardware trade-off without silently shrinking science settings.')
        # Full-resolution inference and all 27 metrics on ONE TRAINING case only.
        old_allowed=set(dataset.allowed);dataset.allowed.add('train')
        from microflow.inference import generate
        from microflow.metrics import evaluate
        source,target,layers=dataset.case(rows[0])
        before=perf_counter();prediction,outside=generate(model,source.to(device),cfg,stats,cfg['seed'])
        values,diag,_,_,_=evaluate(prediction[0,0].cpu().numpy(),target[0,0].numpy(),layers[0].numpy(),dataset.maps(rows[0]),teacher,stats,cfg,device)
        report.update(state='passed',inference_seconds=perf_counter()-before,engineering_metrics=values,engineering_diagnostics=diag,
                      clipped_fraction=outside,full_resolution_inference=True)
        dataset.allowed=old_allowed
        write(dataset.run/'verification/full_size.json',report)
    except Exception as e:
        write(dataset.run/'verification/full_size.json',{'state':'failed','signature':signature,'error':str(e),'formal_updates':0,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)})
        raise
    finally:
        hook.remove()
        optimizer.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True)


def export(run):
    import shutil
    complete=read(run/'training_complete.json');best=complete['best'];test=run/f'test/best_e{best["epoch"]:03d}'
    for relative,expected in complete['signature']['sources'].items():
        if digest(ROOT/relative)!=expected:raise ValueError('Frozen scientific source changed before export: '+relative)
    summary=read(test/'summary.json')
    if summary['patient_count']!=50 or summary['weight_id']!=complete['best_sha256']:raise ValueError('Frozen test delivery is incomplete.')
    out=run/'analysis_return';out.mkdir(exist_ok=True)
    for name in ['normalization.json','split_lock.json','training_complete.json','environment.json','contract.json','config_frozen.json']:
        shutil.copy2(run/name,out/name)
    for folder in ['train','validation','teacher']:
        dest=out/folder;dest.mkdir(exist_ok=True)
        for p in (run/folder).rglob('*.json'):
            target=dest/p.relative_to(run/folder);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    inventory={}
    for p in test.rglob('*'):
        if not p.is_file():continue
        if p.name=='prediction.npy':inventory[str(p.relative_to(run))]={'bytes':p.stat().st_size,'sha256':digest(p)}
        else:
            target=out/'test'/p.relative_to(test);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    write(out/'server_volumes.json',inventory)
    selected_validation=run/f'validation/e{best["epoch"]:03d}'
    for p in selected_validation.rglob('*'):
        if p.is_file():
            target=out/'validation_best'/p.relative_to(selected_validation);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    shutil.copytree(ROOT/'src',out/'source/src',dirs_exist_ok=True);shutil.copy2(ROOT/'run.py',out/'source/run.py')
    shutil.copy2(run/'config_frozen.json',out/'source/main.json')
    write(out/'complete.json',{'patients':50,'best':best,'files':{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='complete.json'}})


def main():
    global ACTIVE_RUN,OWNS_OPERATION
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['info','prepare','teacher','check','fit','validate','test','export','pause','status'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/main.json')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--run',type=Path,default=ROOT/'runs/main')
    parser.add_argument('--device',default=None)
    args=parser.parse_args();cfg=read(args.config);run=args.run.resolve();run.mkdir(parents=True,exist_ok=True)
    tr=cfg['train'];shape=cfg['data']['shape']
    if (tr['patients_per_update'],tr['physical_patient_batch'],tr['gradient_accumulation'],tr['patches_per_patient'])!=(2,1,2,2):
        raise ValueError('This declared implementation uses two patients accumulated sequentially, two patches each.')
    if cfg['patch']['core'][0]!=shape[0] or cfg['patch']['halo'][0]!=0 or cfg['inference']['core'][0]!=shape[0] or cfg['inference']['overlap'][0]!=0:
        raise ValueError('All supervised/inference columns must retain the complete axial ray.')
    if tr['epochs']%tr['validation_every']:raise ValueError('The last epoch must have a committed validation.')
    ACTIVE_RUN=run
    args.device=args.device or ('cpu' if args.command=='prepare' else 'cuda:0')
    if args.command=='pause':(run/'pause.request').touch();print('Pause requested at a complete update/patient boundary.');return
    if args.command=='status':print(json.dumps(read(run/'status.json') if (run/'status.json').exists() else {'phase':'not_started'},indent=2));return
    if args.command=='info':
        from tests.reference_ssm import ReferenceMamba
        with torch.device('meta'):
            model=MicroVesselFlow(cfg['model'],ReferenceMamba)
            from microflow.model import VesselTeacher
            teacher=VesselTeacher(cfg['teacher']['width'],cfg['teacher']['blocks'])
        report={**model.parameter_report(),'teacher':sum(p.numel() for p in teacher.parameters()),'official_cuda_verified':False}
        print(json.dumps(report,indent=2));return
    if args.command=='export':
        with operation_lock(run):
            OWNS_OPERATION=True;export(run)
        return
    if args.manifest is None:parser.error('--manifest is required; import the existing fixed split.')
    if args.command in ('fit','teacher','check','validate','test') and torch.device(args.device).type!='cuda':raise RuntimeError('Formal stages use the official CUDA Mamba operator; CPU is for project tests.')
    with operation_lock(run):
        OWNS_OPERATION=True
        device=torch.device(args.device);env=configure(cfg,device,run);write(run/'environment.json',env)
        if (run/'config_frozen.json').exists() and read(run/'config_frozen.json')!=cfg:
            raise ValueError('Declared experiment configuration changed; create a new numbered experiment.')
        if not (run/'config_frozen.json').exists():write(run/'config_frozen.json',cfg)
        write(run/'environment_history'/('run_'+stamp().replace(':','').replace('+','_')+'.json'),env)
        # Preparing data has no GPU work; user may explicitly choose --device cpu.
        dataset=Dataset(args.manifest,cfg,run,{'test'} if args.command=='test' else {'train','validation'})
        base=source_signature(ROOT,args.config,args.manifest)
        if args.command=='test':
            complete=read(run/'training_complete.json')
            if complete['epochs']!=cfg['train']['epochs'] or complete['updates']!=cfg['train']['epochs']*cfg['train']['patient_groups_per_epoch']:raise ValueError('Full declared budget is required before test.')
            stats=read(run/'normalization.json')
        else:stats=statistics(dataset,cfg)
        base['normalization_sha256']=digest(run/'normalization.json')
        if args.command=='prepare':audit(dataset,cfg);write(run/'prepared.json',{'signature':base,'time':stamp()});return
        if not (run/'prepared.json').exists():raise RuntimeError('Run prepare and inspect annotation overlays before formal stages.')
        if read(run/'prepared.json')['signature']!=base:raise ValueError('Prepared data/source identity changed.')
        if args.command=='test':
            from microflow.model import VesselTeacher
            teacher=VesselTeacher(cfg['teacher']['width'],cfg['teacher']['blocks']).to(device)
            teacher_receipt=read(run/'teacher/complete.json')
            if teacher_receipt['signature']!=base or digest(run/'teacher/best.pt')!=teacher_receipt['sha256']:raise ValueError('Frozen teacher changed.')
            teacher.load_state_dict(load(run/'teacher/best.pt')['model']);teacher.eval().requires_grad_(False)
        else:teacher,teacher_receipt=fit_teacher(dataset,cfg,stats,device,base)
        if args.command=='teacher':return
        signature={**base,'teacher_sha256':teacher_receipt['sha256']};write(run/'contract.json',signature)
        seed_all(cfg['seed']);model=MicroVesselFlow(cfg['model']).to(device)
        if args.command in ('fit','check'):
            check=read(run/'verification/full_size.json') if (run/'verification/full_size.json').exists() else None
            if args.command=='check' or not check or check.get('state')!='passed' or not check.get('full_resolution_inference') or check.get('signature')!=signature or check.get('packages')!=env['packages']:
                gpu_check(model,teacher,dataset,cfg,stats,device,signature)
                if args.command=='check':return
                del model;torch.cuda.empty_cache();seed_all(cfg['seed']);model=MicroVesselFlow(cfg['model']).to(device)
            complete=fit_flow(model,teacher,dataset,cfg,stats,device,signature,teacher_receipt['sha256'])
            dataset=Dataset(args.manifest,cfg,run,{'test'})
        else:complete=read(run/'training_complete.json') if args.command=='test' else None
        if args.command=='validate':
            selected=load(run/'checkpoints/best.pt')
            if selected['signature']!=signature:raise ValueError('Checkpoint contract mismatch.')
            model.load_state_dict(selected['ema'])
            evaluate_split(model,teacher,dataset,'validation',cfg,stats,device,run/'selected_validation',digest(run/'checkpoints/best.pt'),teacher_receipt['sha256']);return
        if complete['signature']!=signature or digest(run/'checkpoints/best.pt')!=complete['best_sha256']:raise ValueError('Frozen best changed.')
        selected=load(run/'checkpoints/best.pt');model.load_state_dict(selected['ema'])
        epoch=complete['best']['epoch'];state(run,'testing',best_epoch=epoch)
        evaluate_split(model,teacher,dataset,'test',cfg,stats,device,run/f'test/best_e{epoch:03d}',complete['best_sha256'],teacher_receipt['sha256'])
        export(run);state(run,'complete',epochs=cfg['train']['epochs'],test_patients=50)


if __name__=='__main__':
    try:main()
    except Pause as e:
        if ACTIVE_RUN is not None and OWNS_OPERATION:
            previous=read(ACTIVE_RUN/'status.json') if (ACTIVE_RUN/'status.json').exists() else {}
            previous.update(phase='paused',time=stamp(),reason=str(e));write(ACTIVE_RUN/'status.json',previous)
        print('Safely paused:',e,flush=True);raise SystemExit(75)
    except Exception as e:
        if ACTIVE_RUN is not None and OWNS_OPERATION:
            previous=read(ACTIVE_RUN/'status.json') if (ACTIVE_RUN/'status.json').exists() else {}
            previous.update(phase='failed',time=stamp(),error_type=type(e).__name__,error=str(e))
            write(ACTIVE_RUN/'status.json',previous)
        raise

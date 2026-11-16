"""Original-size training/inference and resume acceptance, using training cases only."""
import copy
import gc
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from scipy.io import loadmat
from common import ROOT,CONFIG,NETS,read,write,now,digest,source_identity,strict_precision
from common import seed_all,options,build_model,checkpoint,restore,environment,rng_state,restore_rng
from prepare import rasterize


def main():
    assert not (ROOT/'verification/training_launch.json').exists()
    write(ROOT/'verification/preflight.json',dict(status='running',time=now()))
    torch.set_num_threads(8);strict_precision();seed_all(CONFIG['seed'])
    assert torch.__version__=='2.7.0+cu126' and torch.version.cuda=='12.6'
    preparation=read(ROOT/'verification/preparation.json');assert preparation['status']=='passed'
    author_files=read(ROOT/'provenance/author_files.json')
    for relative,expected_hash in author_files.items():
        assert digest(ROOT/'source/author'/relative)==expected_hash, 'Author source bytes changed: '+relative
    membership=read(ROOT/'membership.json');identities=read(ROOT/'verification/data_identity.json')
    eligible={r['case_id'] for r in membership if r['split']!='test'}
    assert set(identities)==eligible and len(identities)==150
    assert preparation['identities_sha256']==digest(ROOT/'verification/data_identity.json')
    checked=0
    print('Checking all 450 derived training/validation input file hashes',flush=True)
    for cid,v in identities.items():
        for relative,info in v['derived'].items():
            path=ROOT/relative;assert path.stat().st_size==info['bytes'] and digest(path)==info['sha256'];checked+=1
        if checked%90==0:print(f'Input SHA256 checks {checked}/450',flush=True)
    # Explicitly verify unknown author preprocessing adaptation with an asymmetric analytic case.
    layers=np.zeros((6,304,304),dtype=np.float32)
    for i,z in enumerate([101,151,201,251,301,401]):layers[i]=z
    layers[:,:,200:]+=25
    seg=rasterize(layers);centers=(np.arange(256)+.5)*640/256-.5
    assert np.array_equal(seg[:,80,20]==1,(centers>=100)&(centers<200))
    assert np.array_equal(seg[:,80,220]==2,(centers>=225)&(centers<=425))
    assert np.array_equal(seg[:,80,20],seg[:,180,20]) and not np.array_equal(seg[:,80,20],seg[:,80,220])
    opt=options()
    # Audit actual epoch-start LambdaLR behavior in the exact training framework.
    from models.networks import get_scheduler
    import warnings
    parameter=torch.nn.Parameter(torch.tensor(0.))
    optimizer=torch.optim.Adam([parameter],lr=CONFIG['learning_rate'])
    scheduler=get_scheduler(optimizer,opt);rates={}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for epoch in range(1,301):
            scheduler.step()
            if epoch in [1,199,200,201,299,300]:rates[str(epoch)]=optimizer.param_groups[0]['lr']
            parameter.grad=torch.ones_like(parameter);optimizer.step()
    assert rates['1']==rates['199']==CONFIG['learning_rate']
    assert math.isclose(rates['200'],CONFIG['learning_rate']*100/101) and rates['300']==0.
    write(ROOT/'verification/scheduler_audit.json',dict(torch=torch.__version__,epoch_learning_rate=rates,original_call_order_preserved=True))
    from data.alignedoct2octaall_dataset import AlignedOCT2OCTAALLDataset
    from evaluate import full_validation,reference_scores
    from util.eval import cal_MAE,cal_PSNR,cal_SSIM
    dataset=AlignedOCT2OCTAALLDataset(opt,'train');sample=dataset[0];cid=sample['pids']
    assert next(r['split'] for r in membership if r['case_id']==cid)=='train'
    batch={k:(v[None] if torch.is_tensor(v) else [v]) for k,v in sample.items()}
    assert batch['A'].shape==(1,1,256,96,96) and batch['seg'].shape==(1,3,256,96,96)
    assert torch.equal(batch['seg'].sum(1),torch.ones_like(batch['A'][:,0]))
    # Independent load/normalization parity with the original stored raw intensity units.
    raw=np.load(dataset.A_3D_paths[0]);expected=torch.from_numpy((raw.astype(np.float64)/255*2-1).astype(np.float32))
    # Full-image author dataset bypasses crop, so this checks all 23,658,496 voxels.
    fullopt=copy.copy(opt);fullopt.preprocess='none';fullsample=AlignedOCT2OCTAALLDataset(fullopt,'train')[0]
    assert torch.equal(fullsample['A'][0],expected)
    del fullsample,expected,raw
    model=build_model(opt);torch.cuda.reset_peak_memory_stats();timings=[]
    print('Starting original-size FP32 D/G updates',flush=True)
    model.update_learning_rate()
    model.set_input(batch)
    for index in range(2):
        begin=time.monotonic();model.optimize_parameters();torch.cuda.synchronize();timings.append(time.monotonic()-begin)
        print(f'Original-size update {index+1} completed in {timings[-1]:.3f} seconds',flush=True)
        assert all(math.isfinite(v) for v in model.get_current_errors().values())
        for name in NETS:
            for p in getattr(model,'net'+name).parameters():
                assert p.dtype==torch.float32 and torch.isfinite(p).all()
                if p.grad is not None:assert p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all()
    training_peak=torch.cuda.max_memory_allocated()
    # Inference is independent of segmentation in the 3D generator; wrapper output agrees exactly.
    with torch.no_grad():
        core=model.netG.generator_3d(model.real_A)[0]
        wrapped=model.netG(model.real_A,model.seg)[0]
        assert torch.equal(core,wrapped)
    del core,wrapped
    temporary=ROOT/'runs/preflight_temporary';temporary.mkdir(parents=True,exist_ok=False)
    identity=source_identity();saved_path=temporary/'resume.pt';best=dict(score=0.,epoch=None)
    checkpoint(model,0,2,best,saved_path,identity);saved_hash=digest(saved_path)
    snapshot=torch.load(saved_path,map_location='cpu',weights_only=False)
    model.optimize_parameters()
    state=restore(model,saved_path,identity)
    assert state==dict(epoch=0,steps=2,best=best)
    for name in NETS:
        for k,v in getattr(model,'net'+name).state_dict().items():assert torch.equal(v.cpu(),snapshot['nets'][name][k])
    for optimizer,old in zip(model.optimizers,snapshot['optimizers']):
        current=optimizer.state_dict();assert current['param_groups']==old['param_groups']
        for key,fields in old['state'].items():
            for field,value in fields.items():
                if torch.is_tensor(value):assert torch.equal(current['state'][key][field].cpu(),value)
                else:assert current['state'][key][field]==value
    assert all(a.state_dict()==b for a,b in zip(model.schedulers,snapshot['schedulers']))
    assert torch.equal(torch.get_rng_state(),snapshot['rng']['torch'])
    assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),snapshot['rng']['cuda']))
    model.optimize_parameters();assert all(math.isfinite(v) for v in model.get_current_errors().values())
    del snapshot
    # Verify author metric arithmetic on controlled nonconstant arrays, including float-safe uint8 conversion.
    a=np.arange(256,dtype=np.float32).reshape(16,16);b=np.clip(a+3,0,255).astype(np.uint8)
    result=reference_scores(a,b)
    for key,fn in [('mae',cal_MAE),('psnr',cal_PSNR),('ssim',cal_SSIM)]:
        assert abs(result[key]-float(fn(a,b,norm=True)))<1e-4
    before=rng_state();torch.cuda.reset_peak_memory_stats();begin=time.monotonic()
    print('Starting full256x304x304 16-window inference and complete metrics on one training patient',flush=True)
    report=full_validation(model,opt,0,phase='train',limit=1,output=ROOT/'verification/preflight_full_volume')
    full_seconds=time.monotonic()-begin
    assert report['cases'][0]['case_id']==cid and report['shape_ZYX']==[256,304,304]
    assert torch.equal(before['torch'],torch.get_rng_state()) and before['python']==__import__('random').getstate()
    assert not hasattr(model,'fake_B_3d')
    assert not list((ROOT/'verification/preflight_full_volume').glob('*.npy'))
    full_peak=torch.cuda.max_memory_allocated()
    shutil.rmtree(temporary)
    assert not temporary.exists()
    write(ROOT/'verification/cleanup.json',dict(time=now(),removed=['runs/preflight_temporary'],
        preserved=['verification/preflight_full_volume','verification/preflight.json','logs/preflight.log'],
        temporary_checkpoint_sha256=saved_hash,archives_created=False,formal_checkpoints_untouched=True))
    write(ROOT/'verification/preflight.json',dict(status='passed',time=now(),source_identity=identity,
        derived_files_sha256_checked=checked,train_case=cid,test_arrays_opened=False,
        training_shape=[1,1,256,96,96],full_shape=[1,1,256,304,304],training_update_seconds=timings,
        training_peak_allocated_bytes=training_peak,full_volume_peak_allocated_bytes=full_peak,
        full_volume_one_training_case_seconds=full_seconds,
        generator_parameters=sum(p.numel() for p in model.netG.parameters()),
        generator_3d_parameters=sum(p.numel() for p in model.netG.generator_3d.parameters()),
        full_resume_optimizer_scheduler_rng_check='passed',full_volume_rng_preserved=True,
        author_source_unchanged=True,environment=environment()))
    print(json.dumps(read(ROOT/'verification/preflight.json'),indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        write(ROOT/'verification/preflight.json',dict(status='failed',time=now(),error=repr(exc),traceback=traceback.format_exc()))
        raise

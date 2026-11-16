"""Mathematical, real training-only GPU, and full-resolution acceptance checks."""
import argparse
import ast
import copy
import io
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import common as c
from cds import CDS,REGIONS,masks,project
import method
from run import binding,load,payload,promote,tensor_digest


def original_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    body = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


class TinyVelocity(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(2,1,1)
        self.calls = 0
        self.last_state = None

    def forward(self, state, time, source):
        self.calls += 1
        self.last_state = state.detach().clone()
        return self.conv(torch.cat((state,source),1))


class TinyPerceptual(nn.Module):
    def perceptual_loss(self, generated, target):
        return F.l1_loss(generated,target)


def synthetic(config):
    torch.set_num_threads(2)
    checks = {}
    # Compare actual retained exp018 generator bytes and time schedule over the full budget.
    assert c.sha(c.ROOT/'fm3d/model.py') == c.sha(c.ROOT/'source/parents/exp018/model.py')
    original = original_functions(c.ROOT/'source/parents/exp018/run.py',{'time_schedule'},dict(np=np))
    for epoch in range(1,201):
        a,mask = c.time_schedule(config,epoch)
        b,bmask = original['time_schedule'](config,epoch)
        assert np.array_equal(a,b) and np.array_equal(mask,bmask)
        assert (a==0).sum()==100 and np.all(a[~mask]>0) and np.all(a<1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        generator = method.make_model(config)
        assert sum(p.numel() for p in generator.parameters())==2097521
        initial = c.model_digest(generator)
    checks['exact_exp018_generator_source_2097521_parameters_and_40000_times'] = True
    layers = np.broadcast_to(np.array([40,90,180,300,420,540],np.float32)[:,None,None],(6,64,64)).copy()
    mask = masks(layers,16,'cpu')
    assert mask.shape==(3,16,64,64) and torch.all(mask.sum(0)==1)
    volume = torch.rand(1,1,16,64,64,requires_grad=True)
    from models.hybrid_generator import UNet3D2DAll
    actual = project(volume,mask)
    for index,name in enumerate(REGIONS):
        expected = UNet3D2DAll.get_projection(None,2*volume-1,mask[index][None,None])
        assert torch.equal(actual[name],expected)
    # Independent original NumPy rasterization catches layer-boundary and axis errors.
    full_layers = np.broadcast_to(layers[:,0:1,0:1],(6,304,304)).copy()
    ref = original_functions(c.ROOT/'source/parent059_prepare.py',{'rasterize'},dict(np=np))
    labels = ref['rasterize'](full_layers,depth=16)[:,:64,:64]
    assert np.array_equal(mask.numpy(),np.stack([labels==1,labels==2,labels==0]))
    gradient = torch.autograd.grad(actual['ILM_OPL'].sum(),volume)[0]
    assert torch.all(gradient[0,0][~mask[0]]==0) and gradient[0,0][mask[0]].abs().min()>0
    signed = torch.full_like(volume,-2.,requires_grad=True)
    project(signed,mask)['ILM_OPL'].abs().mean().backward()
    assert signed.grad.abs().sum()>0
    empty = mask.clone();empty[0]=False
    assert torch.equal(project(volume,empty)['ILM_OPL'],torch.zeros(1,1,64,64))
    checks['author_projection_exact_partition_rasterization_gradients_and_empty_rays'] = True
    before = torch.get_rng_state().clone()
    auxiliary = CDS(config,'cpu',load_vgg=False)
    assert torch.equal(before,torch.get_rng_state())
    auxiliary.vgg = TinyPerceptual()
    checks['auxiliary_initialization_preserves_generator_rng'] = True
    model = TinyVelocity()
    opts = method.optimizers(model,auxiliary,config)
    data = dict(source=torch.rand(1,1,16,64,64),target=torch.rand(1,1,16,64,64),layers=layers,time=0.,origin_ZYX=(0,0,0))
    d_before = tensor_digest(auxiliary.trainable_state())
    values = method.update(model,auxiliary,opts,data,config,diagnostics=True)
    assert values['cds_active']==1 and values['d_updates']==1 and model.calls==1
    assert torch.equal(model.last_state,data['source'])
    assert tensor_digest(auxiliary.trainable_state())!=d_before
    assert all(values['CDS_'+name+'_'+term]>0 for name in REGIONS for term in ('l1','gan','vgg'))
    assert values['cds_output_gradient_rms']>0 and values['fm_output_gradient_rms']>0
    assert math_close(values['cds_loss'],sum(config['cds']['l1_weight']*values['CDS_'+name+'_l1']+
           config['cds']['gan_weight']*values['CDS_'+name+'_gan']+
           config['cds']['vgg_weight']*values['CDS_'+name+'_vgg'] for name in REGIONS))
    checks['single_G_forward_source_only_t0_all_nine_CDS_terms_and_D_updates'] = True
    data['time']=.37;data['layers']=None
    before_d = tensor_digest(auxiliary.trainable_state())
    before_steps = [float(s['step']) for s in opts[1].state.values()]
    values = method.update(model,auxiliary,opts,data,config)
    assert values['cds_active']==values['d_updates']==values['projection_active']==0
    assert before_d==tensor_digest(auxiliary.trainable_state())
    assert before_steps==[float(s['step']) for s in opts[1].state.values()]
    assert values['loss']==values['fm_mse']
    checks['random_time_is_unchanged_FM_with_no_D_update'] = True
    # Compare the actual old update with CDS disabled, including AdamW and clipping.
    old_namespace = original_functions(c.ROOT/'source/parents/exp143/method.py',
                      {'update','endpoint_l1'},dict(np=np,torch=torch,F=F,time=time))
    baseline = TinyVelocity();current = copy.deepcopy(baseline)
    baseline_opt = [torch.optim.AdamW(baseline.parameters(),lr=config['lr'],weight_decay=0.)]
    current_opts = method.optimizers(current,auxiliary,config)
    disabled = copy.deepcopy(config);disabled['cds']['enabled']=False
    for sampled in (0.,.41):
        sample=dict(data,time=sampled,layers=layers if sampled==0 else None)
        old_namespace['update'](baseline,baseline_opt,sample,disabled)
        method.update(current,auxiliary,current_opts,sample,disabled)
        assert c.model_digest(baseline)==c.model_digest(current)
    checks['CDS_off_exact_parent_loss_optimizer_and_clip_parity'] = True
    # Real discriminator optimizer moments, G state and RNG must survive recovery.
    torch_state = c.rng_state()
    buffer = io.BytesIO()
    torch.save(dict(model=model.state_dict(),discriminators=auxiliary.trainable_state(),
                    optimizers=[opt.state_dict() for opt in opts],rng_state=torch_state),buffer)
    data.update(time=0.,layers=layers)
    method.update(model,auxiliary,opts,data,config)
    desired_g,desired_d = c.model_digest(model),tensor_digest(auxiliary.trainable_state())
    buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=False)
    load(saved,model,auxiliary,opts)
    method.update(model,auxiliary,opts,data,config)
    assert c.model_digest(model)==desired_g and tensor_digest(auxiliary.trainable_state())==desired_d
    checks['generator_three_discriminators_two_optimizers_and_RNG_resume'] = True
    # Exercise the real atomic best-promotion recovery using disposable synthetic tensors.
    import run as runner
    c.ROOT.joinpath('verification').mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='synthetic_commit_',dir=c.ROOT/'verification') as directory:
        temporary = Path(directory).resolve()
        assert c.ROOT.resolve() in temporary.parents
        best_path,candidate_path = temporary/'best.pt',temporary/'best_candidate.pt'
        selection = dict(epoch=5,model_tensor_sha256=c.model_digest(model),value=.5)
        c.save_checkpoint(candidate_path,dict(model=model.state_dict(),selection=selection))
        with patch.object(runner,'BEST',best_path),patch.object(runner,'CANDIDATE',candidate_path):
            runner.promote(selection)
            assert best_path.exists() and not candidate_path.exists()
            runner.promote(selection)
        assert tensor_digest(torch.load(best_path,weights_only=False)['model'])==selection['model_tensor_sha256']
    assert not temporary.exists()
    checks['atomic_best_promotion_recovery_and_synthetic_temp_cleanup'] = True
    record = dict(passed=True,time=time.time(),source_contract=c.source_contract(),checks=checks,
                  generator_initial_tensor_sha256=initial,test_arrays_opened=False,
                  pretrained_vgg_used_in_synthetic=False,formal_gpu_check_uses_actual_vgg=True)
    c.write(c.ROOT/'verification/synthetic.json',record)
    print('Synthetic checks passed:',len(checks),'groups',flush=True)
    return record


def math_close(a,b):
    return abs(a-b)<=max(1e-6,1e-6*abs(a))


def gpu(config, full=False):
    if (c.CKPT/'latest.pt').exists():
        raise RuntimeError('Preflight must precede formal initialization; existing training is preserved')
    deployment,rows = binding()
    uuid = subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert deployment['expected_gpu_uuid'] in uuid
    c.configure(config);c.seed_all(config['seed'])
    inputs = c.Inputs(deployment,config)
    model = method.make_model(config).float().to(config['device'])
    initial = c.model_digest(model)
    assert initial==c.read(c.ROOT/'verification/synthetic.json')['generator_initial_tensor_sha256']
    before_cuda = torch.cuda.get_rng_state().clone()
    before_cpu = torch.get_rng_state().clone()
    auxiliary = CDS(config,config['device'])
    assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_cuda,torch.cuda.get_rng_state())
    vgg_path = Path(deployment['torch_home'])/'hub/checkpoints/vgg19-dcbb9e9d.pth'
    assert c.sha(vgg_path)==config['cds']['vgg']['weights_sha256']
    assert not any(p.requires_grad for p in auxiliary.vgg.parameters())
    opts = method.optimizers(model,auxiliary,config)
    training = [r for r in rows if r['split']=='train']
    evidence = []
    torch.cuda.reset_peak_memory_stats()
    for index in (0,17,53,81,103,119,137):
        row = training[index]
        data = method.batch(inputs,row,config,0.)
        assert list(data['source'].shape)==[1,1,256,64,64]
        torch.cuda.synchronize();tick=time.perf_counter()
        value = method.update(model,auxiliary,opts,data,config,diagnostics=index==0)
        torch.cuda.synchronize();seconds=time.perf_counter()-tick
        assert value['cds_active']==value['d_updates']==1
        evidence.append(dict(case_id=row['case_id'],split='train',origin_ZYX=data['origin_ZYX'],
                             measured_seconds=seconds,**value))
        print('GPU training-only preflight',row['case_id'],seconds,'s, CDS=',value['cds_loss'],flush=True)
    row = training[0]
    before_d = tensor_digest(auxiliary.trainable_state())
    value = method.update(model,auxiliary,opts,method.batch(inputs,row,config,.37),config)
    assert value['d_updates']==0 and tensor_digest(auxiliary.trainable_state())==before_d
    assert c.model_digest(model)!=initial
    receipt = dict(passed=True,time=time.time(),source_contract=c.source_contract(),runtime_versions=c.versions(),
                   gpu_uuid=uuid,precision_flags=c.precision_flags(model),evidence=evidence,
                   steady_median_seconds=statistics.median(r['measured_seconds'] for r in evidence[1:]),
                   peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                   peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                   actual_frozen_vgg_verified=True,random_time_no_D_update=True,
                   from_scratch_generator_digest=initial,test_used=False)
    c.write(c.ROOT/'verification/gpu_preflight.json',receipt)
    if full:
        source,target = inputs.pair(row)
        layers,regions = inputs.layers(row['case_id']),inputs.regions_xy(row['case_id'])
        before = c.model_digest(model)
        started = time.perf_counter()
        prediction = method.predict(model,torch.from_numpy(np.array(source,copy=True))[None],config)
        assert prediction.shape==(1,1,256,304,304) and prediction.dtype==torch.float32
        pred,gt = prediction[0,0].numpy(),target[0]
        metrics,p,g,slices = c.quality(pred,gt,layers,regions,include_reference=True)
        assert np.isfinite(pred).all() and len(slices)==304
        assert c.model_digest(model)==before
        output = c.ROOT/'verification/full_training_case'
        output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/'projections.npz',**{'pred_'+k:v for k,v in p.items()},**{'gt_'+k:v for k,v in g.items()})
        c.case_figure(output/'comparison.png',row['case_id'],p,g,pred,gt)
        c.write(output/'metrics.json',dict(case_id=row['case_id'],split='train',metrics=metrics,bscan_count=304,
                                         complete_shape=[256,304,304],volume_saved=False))
        c.write(c.ROOT/'verification/full_preflight.json',dict(passed=True,time=time.time(),
                source_contract=c.source_contract(),case_id=row['case_id'],split='train',test_used=False,
                shape=[256,304,304],full_euler_steps=8,volume_saved=False,seconds=time.perf_counter()-started,
                model_unchanged_during_inference=True,metric_protocol=c.METRIC_PROTOCOL))
        print('Full training-case Euler8 and all metric groups passed',flush=True)
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['synthetic','gpu','all'])
    args=parser.parse_args();config=c.read(c.ROOT/'experiment.json')
    if args.action in ('synthetic','all'): synthetic(config)
    if args.action in ('gpu','all'): gpu(config,full=args.action=='all')

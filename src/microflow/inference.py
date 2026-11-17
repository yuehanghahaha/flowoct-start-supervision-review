"""Synchronous whole-state ODE: blend velocities, never independent tile paths."""
from itertools import product
import torch
from .runtime import amp
from .geometry import starts,extract,sample_context,region,core_slice,blend_window
from .flow import reference,noise_for


def whole_velocity(model,source,state,time,cfg):
    core=cfg['inference']['core'];halo=cfg['inference']['halo'];overlap=cfg['inference']['overlap']
    shape=state.shape[-3:];times=state.new_full((1,),time)
    _,q=reference(state,times)
    with amp(state.device):context=model.encode_context(source,state/q.sqrt(),times)
    result=torch.zeros_like(state);denominator=torch.zeros_like(state)
    for pos in product(*(starts(n,c,o) for n,c,o in zip(shape,core,overlap))):
        oct_p=extract(source,pos,core,halo);state_p=extract(state,pos,core,halo)
        context_p=sample_context(context,pos,core,halo,shape)
        ref,qp=reference(state_p,times)
        with amp(state.device):residual=model.forward_patch(oct_p,state_p/qp.sqrt(),times,context_p)
        velocity=ref+residual.float()
        cuts=core_slice(pos,core,halo,shape);dst=region(pos,core,shape)
        value=velocity[(slice(None),slice(None),*cuts)]
        window=blend_window(value.shape[-3:],state.device)[None,None]
        index=(slice(None),slice(None),*dst)
        result[index]+=window*value;denominator[index]+=window
    if (denominator<=0).any():raise RuntimeError('Inference has uncovered voxels.')
    return result/denominator


@torch.inference_mode()
def generate(model,source,cfg,stats,seed):
    model.eval();state=noise_for(source.shape,source.device,seed);steps=cfg['inference']['steps']
    for i in range(steps):
        t=i/steps;dt=1/steps
        first=whole_velocity(model,source,state,t,cfg)
        second=whole_velocity(model,source,state+dt*first,t+dt,cfg)
        state=state+dt*.5*(first+second)
        if not torch.isfinite(state).all():raise FloatingPointError('Nonfinite whole-volume trajectory.')
    raw=state*stats['volume_rms']
    outside=float(((raw<0)|(raw>1)).float().mean())
    return raw.clamp(0,1),outside

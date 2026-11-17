import torch
from .runtime import amp
from .geometry import extract,sample_context,region,core_slice,slab


def reference(state,time):
    t=time[:,None,None,None,None].float();q=(1-t).square()+t.square()
    return (2*t-1)*state/q,q


def planes(array,start,core,halo):
    out=array
    for axis,s,c,h,n in [(-2,start[1],core[1],halo[1],array.shape[-2]),(-1,start[2],core[2],halo[2],array.shape[-1])]:
        out=out.index_select(axis,torch.arange(s-h,s+c+h,device=out.device).clamp(0,n-1))
    return out


def patches(model,source,state,time,positions,core,halo):
    _,q=reference(state,time)
    with amp(state.device):context=model.encode_context(source,state/q.sqrt(),time)
    oct_p=torch.cat([extract(source,s,core,halo) for s in positions])
    state_p=torch.cat([extract(state,s,core,halo) for s in positions])
    context_p=torch.cat([sample_context(context,s,core,halo,state.shape[-3:]) for s in positions])
    times=time.expand(len(positions))
    _,qp=reference(state_p,times)
    with amp(state.device):r=model.forward_patch(oct_p,state_p/qp.sqrt(),times,context_p)
    vref,_=reference(state_p,times)
    return r.float()+vref,state_p


def noise_for(shape,device,seed):
    g=torch.Generator(device=device).manual_seed(seed)
    return torch.randn(shape,device=device,dtype=torch.float32,generator=g)

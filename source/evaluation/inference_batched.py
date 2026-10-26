"""Same voxel Euler/Hann protocol; independent windows may share a forward batch."""
from itertools import product
import torch
from torch.nn import functional as F
from fm3d.inference import window_starts


@torch.inference_mode()
def generate_batched(model, source, patch, steps=8, overlap=.5, batch_size=1, progress=None):
    assert source.ndim==5 and source.shape[:2]==(1,1)
    assert batch_size>=1 and steps>=1 and 0<=overlap<1
    device=next(model.parameters()).device
    original=source.shape[2:]
    pad=[]
    for n,p in reversed(list(zip(original,patch))):pad.extend([0,max(n,p)-n])
    source=F.pad(source.detach().to('cpu',dtype=torch.float32),pad,mode='replicate')
    starts=[window_starts(n,p,overlap) for n,p in zip(source.shape[2:],patch)]
    regions=[(slice(None),slice(None))+tuple(slice(s,s+p) for s,p in zip(pos,patch)) for pos in product(*starts)]
    axes=[torch.hann_window(p,periodic=False).clamp_min(.05) for p in patch]
    weight=(axes[0][:,None,None]*axes[1][None,:,None]*axes[2][None,None,:])[None,None]
    counts=torch.zeros_like(source)
    for region in regions:counts[region]+=weight
    state=source.clone();was_training=model.training;model.eval()
    try:
        for step in range(steps):
            velocity=torch.zeros_like(state)
            for first in range(0,len(regions),batch_size):
                group=regions[first:first+batch_size]
                current=torch.cat([state[r] for r in group],dim=0).to(device)
                condition=torch.cat([source[r] for r in group],dim=0).to(device)
                times=torch.full((len(group),),step/steps,device=device)
                v=model(current,times,condition)
                if not torch.isfinite(v).all():raise FloatingPointError('Nonfinite velocity')
                v=v.float().cpu()
                # Keep region accumulation order exactly the same as the original.
                for i,region in enumerate(group):velocity[region]+=v[i:i+1]*weight
            state.add_(velocity/counts,alpha=1/steps)
            if progress:progress(step+1,steps,len(regions))
        return state[:,:,:original[0],:original[1],:original[2]].clamp(0,1)
    finally:model.train(was_training)

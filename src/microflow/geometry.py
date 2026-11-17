"""Whole-depth local columns; every supervised MIP contains the entire slab."""
from itertools import product
import numpy as np
import torch
from torch.nn import functional as F


def starts(n, width, overlap=0):
    if not 0<=overlap<width: raise ValueError('Invalid overlap.')
    if n<=width:return [0]
    values=list(range(0,n-width+1,width-overlap))
    if values[-1]!=n-width:values.append(n-width)
    return values


def uniform_tile(shape, core, rng):
    tiles=list(product(*(range(0,n,c) for n,c in zip(shape,core))))
    counts=np.array([np.prod([min(c,n-s) for n,c,s in zip(shape,core,pos)]) for pos in tiles],float)
    return tuple(int(v) for v in tiles[int(rng.choice(len(tiles),p=counts/counts.sum()))])


def extract(volume, start, core, halo):
    # Clamped indexing gives fixed shapes at edges without violating memory bounds.
    out=volume
    for axis,(s,c,h,n) in enumerate(zip(start,core,halo,volume.shape[-3:]),start=volume.ndim-3):
        index=torch.arange(s-h,s+c+h,device=volume.device).clamp(0,n-1)
        out=out.index_select(axis,index)
    return out


def core_slice(start, core, halo, shape):
    return tuple(slice(h,h+min(c,n-s)) for s,c,h,n in zip(start,core,halo,shape))


def region(start, core, shape):
    return tuple(slice(s,min(s+c,n)) for s,c,n in zip(start,core,shape))


def sample_context(context, start, core, halo, full_shape):
    coords=[]
    for s,c,h,n in zip(start,core,halo,full_shape):
        index=torch.arange(s-h,s+c+h,device=context.device).float().clamp(0,n-1)
        coords.append(2*(index+.5)/n-1)
    zz,yy,xx=torch.meshgrid(*coords,indexing='ij')
    grid=torch.stack((xx,yy,zz),-1)[None].expand(context.shape[0],-1,-1,-1,-1)
    # grid_sample requires the grid and input to have the same dtype outside autocast.
    return F.grid_sample(context.float(),grid,mode='bilinear',padding_mode='border',align_corners=False)


def slab(volume, layers, lower=0, upper=2, native_depth=640, mode='max'):
    z=(torch.arange(volume.shape[2],device=volume.device).float()+.5)*native_depth/volume.shape[2]-.5
    support=(z[None,:,None,None]>=layers[:,lower:lower+1]-1)&(z[None,:,None,None]<=layers[:,upper:upper+1]-1)
    count=support.sum(1);valid=count>0
    x=volume[:,0]
    if mode=='max':out=x.masked_fill(~support,-torch.finfo(x.dtype).max).amax(1)
    elif mode=='mean':out=(x*support).sum(1)/count.clamp_min(1)
    else:raise ValueError(mode)
    return torch.where(valid,out,0)[:,None],valid[:,None]


def blend_window(shape,device):
    w=torch.ones(shape,device=device)
    for axis,n in enumerate(shape):
        part=torch.hann_window(n+2,periodic=False,device=device)[1:-1].clamp_min(.01)
        view=[1]*3;view[axis]=n;w=w*part.reshape(view)
    return w

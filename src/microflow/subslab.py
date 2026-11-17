"""Layer-relative, nonoverlapping max projections; no 3D vessel labels.

These projections detect some axial misplacement, but retain ambiguity inside
each bin. Native layer coordinates use the same voxel-center mapping as slab().
"""
import torch


def relative_subslabs(volume, layers, bins=4, lower=0, upper=2, native_depth=640):
    """Return projections and valid rays, both [B,K,Y,X].

    Per A-line, partition [lower,upper] into equal relative-depth bins. Bins
    are left-closed/right-open except the final closed bin. A voxel belongs
    to at most one bin. Thin bins with no voxel centers are invalid, not zero
    intensity targets. Invalid/collapsed layer intervals are also excluded.
    """
    if isinstance(bins,bool) or not isinstance(bins,int) or bins<1:
        raise ValueError('Sub-slab bin count must be a positive integer.')
    if volume.ndim!=5 or volume.shape[1]!=1 or not volume.is_floating_point():
        raise ValueError('Expected a floating-point [B,1,Z,Y,X] volume.')
    if (layers.ndim!=4 or layers.shape[0]!=volume.shape[0] or
        layers.shape[-2:]!=volume.shape[-2:] or not 0<=lower<upper<layers.shape[1]):
        raise ValueError('Layer geometry does not match the volume.')
    if native_depth<=0 or layers.device!=volume.device:
        raise ValueError('Invalid native depth or mismatched geometry device.')
    # Geometry and intensity reductions stay float32 outside network autocast.
    x=volume[:,0].float()
    lo=layers[:,lower:lower+1].float()-1
    hi=layers[:,upper:upper+1].float()-1
    valid_line=torch.isfinite(lo)&torch.isfinite(hi)&(hi>lo)
    z=(torch.arange(x.shape[1],device=x.device,dtype=torch.float32)+.5)*native_depth/x.shape[1]-.5
    z=z[None,:,None,None]
    projections=[];valid_rays=[]
    for k in range(bins):
        left=lo+(hi-lo)*(k/bins)
        right=hi if k==bins-1 else lo+(hi-lo)*((k+1)/bins)
        support=valid_line&(z>=left)&((z<=right) if k==bins-1 else (z<right))
        valid=support.any(1)
        # max saves indices rather than K full masked volumes for backward.
        # Tied maxima use the first valid axial location as a subgradient.
        projection=x.masked_fill(~support,-torch.inf).max(1).values
        projections.append(torch.where(valid,projection,0.))
        valid_rays.append(valid)
    return torch.stack(projections,1),torch.stack(valid_rays,1)


def subslab_l1(prediction, target, valid):
    """Equal valid-bin mean per patient, then equal patient mean (raw units).

    Each bin averages only valid A-lines. Empty bins are excluded; a patient
    with no valid bins contributes a differentiable zero. No vessel labels,
    teacher probabilities, bin-specific intensity stretching or GT dilation.
    """
    if prediction.shape!=target.shape or prediction.shape!=valid.shape or prediction.ndim!=4:
        raise ValueError('Expected matching [B,K,Y,X] projections and masks.')
    weight=valid.to(dtype=torch.float32)
    count=weight.sum((-2,-1))
    error=((prediction.float()-target.float()).abs()*weight).sum((-2,-1))/count.clamp_min(1)
    available=(count>0).float()
    return ((error*available).sum(1)/available.sum(1).clamp_min(1)).mean()

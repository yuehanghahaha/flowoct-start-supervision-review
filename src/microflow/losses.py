"""Vessel losses act through the generated image and a frozen image classifier.

clDice-inspired differentiable skeletonization is reimplemented here. The
precomputed GT centerline uses the full 2D annotation before patch extraction.
This is a projection topology surrogate, not a 3D homotopy guarantee.
"""
import torch
from torch.nn import functional as F
from .runtime import amp
from .geometry import extract,core_slice,slab
from .flow import patches,planes
from .subslab import relative_subslabs,subslab_l1


def soft_skeleton(p,iterations):
    def erode(x):return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),-F.max_pool2d(-x,(1,3),1,(0,1)))
    def opening(x):return F.max_pool2d(erode(x),3,1,1)
    skel=F.relu(p-opening(p))
    for _ in range(iterations):
        p=erode(p);delta=F.relu(p-opening(p));skel=skel+F.relu(delta-skel*delta)
    return skel


def cldice(p,truth,skeleton,pred_skeleton,valid):
    eps=1e-6
    pred_skeleton=pred_skeleton*valid;skeleton=skeleton*valid
    precision=((pred_skeleton*truth).sum()+eps)/(pred_skeleton.sum()+eps)
    sensitivity=((skeleton*p).sum()+eps)/(skeleton.sum()+eps)
    return 1-2*precision*sensitivity/(precision+sensitivity+eps)


def patient_objective(model,teacher,source,target,layers,maps,z0,time,positions,cfg,stats,step,ramp_override=None):
    core=cfg['patch']['core'];halo=cfg['patch']['halo'];shape=source.shape[-3:]
    scale=stats['volume_rms'];t=time[:,None,None,None,None]
    z1=target/scale;state=(1-t)*z0+t*z1
    velocity,state_p=patches(model,source,state,time,positions,core,halo)
    endpoint=(state_p+(1-t)*velocity)*scale
    truth_p=torch.cat([extract(target,s,core,halo) for s in positions])
    wanted_p=torch.cat([extract(z1-z0,s,core,halo) for s in positions])
    layer_p=torch.cat([planes(layers,s,core,halo) for s in positions])
    mip,valid=slab(endpoint,layer_p,native_depth=cfg['data']['native_layer_depth'])
    mip_truth,_=slab(truth_p,layer_p,native_depth=cfg['data']['native_layer_depth'])
    sub_options={**cfg['subslab'],'native_depth':cfg['data']['native_layer_depth']}
    with amp(source.device):logits=teacher(mip/stats['mip_rms'])
    probabilities=logits.float().softmax(1)
    maps_p={k:torch.cat([planes(torch.as_tensor(v,device=source.device)[None,None].float(),s,core,halo) for s in positions])
            for k,v in maps.items() if hasattr(v,'shape')}
    pred_skel=soft_skeleton(probabilities[:,1:2],cfg['loss']['skeleton_iterations'])
    values={k:[] for k in ('endpoint','weighted_mip','vessel_ce','cldice','gradient_match','subslab')}
    for i,start in enumerate(positions):
        cuts=core_slice(start,core,halo,shape);roi=(slice(i,i+1),slice(None),*cuts)
        if i==0:fm=F.mse_loss(velocity[roi],wanted_p[roi])
        values['endpoint'].append(F.l1_loss(endpoint[roi],truth_p[roi])/scale)
        grad=[]
        for axis in (2,3,4):
            if endpoint[roi].shape[axis]>1:
                grad.append(F.l1_loss(torch.diff(endpoint[roi],dim=axis),torch.diff(truth_p[roi],dim=axis))/scale)
        values['gradient_match'].append(sum(grad)/max(1,len(grad)))
        xy=(slice(i,i+1),slice(None),cuts[1],cuts[2]);v=valid[xy].float()
        # Projection is independent across A-lines. Use the valid core and its
        # aligned boundaries directly, with the original full axial extent.
        sub_pred,sub_valid=relative_subslabs(endpoint[roi],layer_p[xy],**sub_options)
        sub_truth,_=relative_subslabs(truth_p[roi],layer_p[xy],**sub_options)
        values['subslab'].append(subslab_l1(sub_pred,sub_truth,sub_valid)/stats['mip_rms'])
        cap=maps_p['capillary'][xy];large=maps_p['large'][xy];faz=maps_p['faz'][xy]
        weights=torch.where(cap>0,cfg['loss']['capillary_weight'],1.)
        weights=torch.where(maps_p['medium_centerline'][xy]>0,cfg['loss']['medium_centerline_weight'],weights)
        weights=torch.where(maps_p['small_centerline'][xy]>0,cfg['loss']['small_centerline_weight'],weights)
        weights=torch.where(faz>0,cfg['loss']['faz_weight'],weights)
        weights=weights*v
        values['weighted_mip'].append(((mip[xy]-mip_truth[xy]).abs()*weights).sum()/weights.sum().clamp_min(1)/stats['mip_rms'])
        labels=cap[:,0].long()+2*large[:,0].long()
        ce=F.cross_entropy(logits[i:i+1,:,cuts[1],cuts[2]].float(),labels,reduction='none')
        ce_weight=(1+2*cap[:,0])*v[:,0]
        values['vessel_ce'].append((ce*ce_weight).sum()/ce_weight.sum().clamp_min(1))
        values['cldice'].append(cldice(probabilities[:,1:2][xy],cap,maps_p['skeleton'][xy],pred_skel[xy],v))
    components={k:torch.stack(value).mean() for k,value in values.items()}
    ramp=min(1.,step/cfg['loss']['ramp_updates']) if ramp_override is None else ramp_override
    total=cfg['loss']['fm_uniform']*fm+ramp*sum(cfg['loss'][k]*value for k,value in components.items())
    return total,{'fm_uniform':fm,**components,'total':total,'ramp':source.new_tensor(ramp)}


def teacher_loss(logits,labels):
    ce=F.cross_entropy(logits.float(),labels,weight=logits.new_tensor([1.,3.,1.]).float())
    p=logits.float().softmax(1)
    dice=[]
    for c in (1,2):
        truth=(labels==c).float();pred=p[:,c]
        dice.append(1-(2*(pred*truth).sum()+1)/(pred.sum()+truth.sum()+1))
    return ce+.5*sum(dice)

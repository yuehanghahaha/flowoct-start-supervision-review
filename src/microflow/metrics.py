"""Standard reconstruction metrics plus separately labelled vascular diagnostics."""
import numpy as np
import torch
from scipy.ndimage import binary_dilation
from skimage.metrics import structural_similarity
from skimage.morphology import skeletonize
from .geometry import slab
from .runtime import amp
from .subslab import relative_subslabs


METRICS=('mae','psnr','ssim')
GROUPS=('volume','bscan2d','ilm_opl_max','opl_bm_max','full_mean','full_max','ilm_opl_mean','opl_bm_mean','ilm_bm_mean')
PROTOCOL={'intensity':'stored /255, fixed [0,1], no per-image stretching','data_range':1.,
 'volume_ssim':'3D','bscan':'all Y-indexed ZX slices; average slices within patient then patients',
 'projection_ssim':'2D','ssim_window':7,'gaussian_weights':False,'sample_covariance':True,
 'aggregation':'equal patient macro mean','slabs':'native640 one-based boundaries; closed voxel-center intervals',
 'projection_orientation':'YX; annotation transform explicitly declared',
 'diagnostics':'2D label-based and teacher-based, not independent 3D capillary ground truth',
 'subslab':'additional GT-intensity depth diagnostic; disjoint relative intervals with closed final edge; raw MAE over valid rays, equal nonempty-bin mean per patient'}


def scores(p,y):
    error=p.astype(np.float64)-y
    return {'mae':float(np.abs(error).mean()),'psnr':float(-10*np.log10(max(np.square(error).mean(),1e-12))),
            'ssim':float(structural_similarity(y,p,data_range=1.,win_size=7,gaussian_weights=False,use_sample_covariance=True,channel_axis=None))}


def projections(volume,layers,native_depth):
    x=torch.as_tensor(volume)[None,None];l=torch.as_tensor(layers)[None]
    out={'full_mean':volume.mean(0),'full_max':volume.max(0)}
    for name,lo,hi,mode in [('ilm_opl_max',0,2,'max'),('opl_bm_max',2,5,'max'),
        ('ilm_opl_mean',0,2,'mean'),('opl_bm_mean',2,5,'mean'),('ilm_bm_mean',0,5,'mean')]:
        out[name]=slab(x,l,lo,hi,native_depth,mode)[0][0,0].numpy()
    return out


def masked_mae(pred,truth,mask):
    return float(np.abs(pred[mask]-truth[mask]).mean()) if mask.any() else None


def depth_diagnostics(pred,truth,layers,cfg):
    options={**cfg['subslab'],'native_depth':cfg['data']['native_layer_depth']}
    geometry=torch.as_tensor(layers)[None]
    p,valid=relative_subslabs(torch.as_tensor(pred)[None,None],geometry,**options)
    y,_=relative_subslabs(torch.as_tensor(truth)[None,None],geometry,**options)
    p=p[0].numpy();y=y[0].numpy();valid=valid[0].numpy()
    diag={};images_p={};images_y={};available=[]
    for k in range(cfg['subslab']['bins']):
        name=f'subslab_{k+1:02d}'
        error=masked_mae(p[k],y[k],valid[k])
        diag[name+'_mae_raw']=error
        diag[name+'_valid_ray_fraction']=float(valid[k].mean())
        if error is not None:available.append(error)
        images_p[name]=p[k];images_y[name]=y[k]
        images_p[name+'_valid']=valid[k];images_y[name+'_valid']=valid[k]
    diag['subslab_mae_raw']=float(np.mean(available)) if available else None
    diag['subslab_nonempty_bins']=len(available)
    return diag,images_p,images_y


@torch.inference_mode()
def evaluate(pred,truth,layers,maps,teacher,stats,cfg,device):
    values={'volume':scores(pred,truth)}
    slices=[{'y':i,**scores(pred[:,i],truth[:,i])} for i in range(truth.shape[1])]
    values['bscan2d']={m:float(np.mean([r[m] for r in slices])) for m in METRICS}
    p=projections(pred,layers,cfg['data']['native_layer_depth']);y=projections(truth,layers,cfg['data']['native_layer_depth'])
    values.update({name:scores(p[name],y[name]) for name in GROUPS[2:]})
    mip=p['ilm_opl_max'];reference=y['ilm_opl_max']
    valid=slab(torch.as_tensor(truth)[None,None],torch.as_tensor(layers)[None],native_depth=cfg['data']['native_layer_depth'])[1][0,0].numpy()
    diag={name+'_mae':masked_mae(mip,reference,maps[key]&valid) for name,key in
          [('capillary','capillary'),('capillary_skeleton','skeleton'),('background','background'),('large','large')]}
    diag['volume_mae']=values['volume']['mae']
    weights=cfg['selection']['weights'];available=[k for k in weights if diag[k] is not None]
    selection=sum(weights[k]*diag[k] for k in available)/sum(weights[k] for k in available)
    image=torch.as_tensor(mip,device=device)[None,None]/stats['mip_rms']
    with amp(device):classes=teacher(image).argmax(1)[0].cpu().numpy()
    cap=(classes==1)&valid;truth_cap=maps['capillary']&valid;tp=int((cap&truth_cap).sum());fp=int((cap&~truth_cap).sum());fn=int((~cap&truth_cap).sum())
    pred_skeleton=skeletonize(cap);true_skeleton=maps['skeleton']&valid
    prec=float((pred_skeleton&truth_cap).sum()/max(1,pred_skeleton.sum()))
    recall=float((true_skeleton&cap).sum()/max(1,true_skeleton.sum()))
    diag.update(teacher_capillary_precision=tp/max(1,tp+fp),teacher_capillary_recall=tp/max(1,tp+fn),
        teacher_capillary_dice=2*tp/max(1,2*tp+fp+fn),teacher_cldice=2*prec*recall/max(1e-12,prec+recall),
        teacher_faz_false_positive=float((classes[maps['faz']]>0).mean()) if maps['faz'].any() else None,
        small_centerline_recall_1px=float(binary_dilation(cap)[maps['small_centerline']].mean()) if maps['small_centerline'].any() else None,
        diagnostic_uses_training_teacher=True,apparent_width_units='pixel',selection_score=selection)
    depth,depth_p,depth_y=depth_diagnostics(pred,truth,layers,cfg)
    diag.update(depth);p.update(depth_p);y.update(depth_y)
    return values,diag,slices,p,y

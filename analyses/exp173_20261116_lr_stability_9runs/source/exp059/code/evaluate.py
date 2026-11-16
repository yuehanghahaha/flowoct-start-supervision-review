"""Full-resolution supplemental evaluation; never retain intermediate 3D arrays."""
import copy
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from common import ROOT, write, now, preserve_rng
from data.alignedoct2octaall_dataset import AlignedOCT2OCTAALLDataset
from util.eval import cal_MAE, cal_MSE, cal_PSNR, cal_SSIM, cal_Perceptual
from util.eval import extract_full_proj_from_volume, extract_proj_from_volume


def numeric(value):
    value=float(value)
    return value if math.isfinite(value) else None


def fixed_scores(gt,pred):
    gt=np.asarray(gt,dtype=np.float32);pred=np.asarray(pred,dtype=np.float32)
    error=gt-pred
    return dict(mae=numeric(np.abs(error).mean(dtype=np.float64)),
                mse=numeric((error.astype(np.float64)**2).mean()),
                psnr=numeric(peak_signal_noise_ratio(gt,pred,data_range=1.)),
                ssim=numeric(structural_similarity(gt,pred,data_range=1.)))


def reference_scores(gt,pred,perceptual=None,device='cuda:0'):
    gt=np.asarray(gt,dtype=np.float32);pred=np.asarray(pred,dtype=np.float32)
    if gt.max()==gt.min():
        return dict(mae=None,mse=None,psnr=None,ssim=None,undefined='constant_GT_range')
    result={name:numeric(fn(gt,pred,norm=True)) for name,fn in
            [('mae',cal_MAE),('mse',cal_MSE),('psnr',cal_PSNR),('ssim',cal_SSIM)]}
    if perceptual is not None:
        a=torch.from_numpy(gt.copy())[None,None].to(device)
        b=torch.from_numpy(pred.copy())[None,None].to(device)
        result['perceptual']=numeric(cal_Perceptual(perceptual,a,b,norm=True))
    return result


def projections(volume,seg):
    return {'full':extract_full_proj_from_volume(volume,seg,'avg'),
            'mean':volume.mean(0),
            'ilm_opl':extract_proj_from_volume(volume,seg,1,'avg'),
            'opl_bm':extract_proj_from_volume(volume,seg,2,'avg')}


def figure(path,case,gt,pred):
    canvas=Image.new('RGB',(660,1430),'white');draw=ImageDraw.Draw(canvas)
    draw.text((12,10),f'{case} | GT / XOCT | same GT window per row',fill='black')
    for row,name in enumerate(['full','mean','ilm_opl','opl_bm']):
        y=40+row*345;draw.text((12,y),name+' (layer-weighted mean; mean = all depth)',fill='black')
        lo=float(gt[name].min());hi=float(gt[name].max())
        for col,a in enumerate([gt[name],pred[name]]):
            a=np.clip((a-lo)/max(hi-lo,1e-12),0,1)
            im=Image.fromarray(np.rint(a*255).astype(np.uint8))
            canvas.paste(im,(12+col*328,y+24))
    canvas.save(path)


def macro(rows,key):
    groups={}
    for r in rows:
        groups.setdefault(r['patient_id'],[]).append(r[key])
    names=set.intersection(*(set(r[key]) for r in rows))
    return {k:float(np.mean([np.mean([v[k] for v in values if isinstance(v.get(k),(float,int))])
                            for values in groups.values() if any(isinstance(v.get(k),(float,int)) for v in values)]))
            for k in sorted(names) if all(isinstance(r[key].get(k),(float,int)) for r in rows)}


def full_validation(model,opt,epoch,phase='val',limit=None,output=None):
    assert phase in ['train','val'], 'Formal test is separate and must be frozen before any test preparation'
    folder=Path(output or ROOT/'reports/validation'/f'epoch_{epoch:03d}')
    if (folder/'metrics.json').exists():
        record=__import__('json').loads((folder/'metrics.json').read_text('utf-8'))
        assert record['epoch']==epoch and record['phase']==phase
        return record
    folder.mkdir(parents=True,exist_ok=True)
    settings=copy.copy(opt);settings.preprocess='none';dataset=AlignedOCT2OCTAALLDataset(settings,phase)
    membership={r['case_id']:r for r in __import__('json').loads((ROOT/'membership.json').read_text('utf-8'))}
    rows=[]
    with preserve_rng(),torch.no_grad():
        for index in range(len(dataset) if limit is None else min(limit,len(dataset))):
            sample=dataset[index];cid=sample['pids'];assert membership[cid]['split']!='test'
            data={k:(v[None] if torch.is_tensor(v) else [v]) for k,v in sample.items()}
            model.sliding_window_aggregation(data,[4,4],[96,96])
            pred=np.clip((model.fake_B_3d[0,0].cpu().numpy()+1)/2,0,1)
            assert pred.shape==(256,304,304) and np.isfinite(pred).all()
            # Read unnormalized raw GT, exactly as the author's evaluation entrypoint.
            gt=np.load(dataset.B_3D_paths[index],allow_pickle=False)
            seg=np.load(dataset.seg_paths[index],allow_pickle=False)
            quantized=(pred*255).astype(np.uint8)
            g=projections(gt,seg);q=projections(quantized,seg);f=projections(pred,seg)
            record=dict(case_id=cid,patient_id=membership[cid]['patient_id'],
                        reference_volume=reference_scores(gt,quantized),
                        fixed_volume=fixed_scores(gt/255,pred))
            for name in g:
                record['reference_'+name]=reference_scores(g[name],q[name],model.criterionPerceptual)
                record['fixed_'+name]=fixed_scores(g[name]/255,f[name])
            for label,name in [(1,'ilm_opl'),(2,'opl_bm')]:
                gm=np.where(seg==label,gt/255,0).max(0)
                pm=np.where(seg==label,pred,0).max(0)
                record['diagnostic_max_'+name]=fixed_scores(gm,pm)
            np.savez_compressed(folder/f'{cid}_projections.npz',
                **{prefix+'_'+k:v.astype(np.float32) for prefix,maps in
                   [('gt',g),('prediction_float',f),('prediction_uint8',q)] for k,v in maps.items()})
            figure(folder/f'{cid}.png',cid,g,{k:v*255 for k,v in f.items()})
            rows.append(record)
            print(f'Full validation epoch {epoch}: {len(rows)}/{len(dataset) if limit is None else limit} {cid}',flush=True)
            del sample,data,pred,gt,seg,quantized,g,q,f,model.fake_B_3d
    groups=['reference_volume','fixed_volume']+['reference_'+k for k in ['full','mean','ilm_opl','opl_bm']]
    groups+=['fixed_'+k for k in ['full','mean','ilm_opl','opl_bm']]+['diagnostic_max_ilm_opl','diagnostic_max_opl_bm']
    result=dict(epoch=epoch,phase=phase,time=now(),shape_ZYX=[256,304,304],cases=rows,
        patient_macro={k:macro(rows,k) for k in groups},intermediate_3d_saved=False,
        reference_protocol='uint8 prediction; GT-range normalization and clipping separately per patient/object; MAE in 0..255 units; PSNR/SSIM data_range=normalized GT range; scalar arithmetic in float32',
        fixed_protocol='float32 predictions; fixed /255 GT; data_range=1; volume SSIM in 3D, projection SSIM in 2D; patient equal-weight macro',
        projection_protocol='full=ILM-BM mask-weighted mean; mean=all-depth mean; ilm_opl/opl_bm=mask-weighted mean; maximum slabs explicitly diagnostic',
        missing_reference_metric='3D Perp.: no public implementation or specified slice aggregation; not invented',
        zero_or_infinite_metric_policy='undefined quantities stored null, never replaced with invented finite numbers')
    write(folder/'metrics.json',result)
    return result

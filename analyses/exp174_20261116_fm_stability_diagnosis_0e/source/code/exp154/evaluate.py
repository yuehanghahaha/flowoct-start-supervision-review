"""Shared full-volume validation on the retained XOCT grid."""
import csv
import math

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
import torch

from common import CDS, ROOT, model_digest, now, read, write
from inputs import normalize
from util.eval import cal_MAE, cal_PSNR, cal_SSIM, cal_Perceptual
from util.eval import extract_full_proj_from_volume, extract_proj_from_volume


def score(gt, pred):
    # Accumulate errors in chunks. skimage evaluates the complete object.
    absolute = squared = 0.
    for first in range(0, gt.shape[0], 16):
        delta = pred[first:first+16].astype(np.float64) - gt[first:first+16]
        absolute += np.abs(delta).sum()
        squared += np.square(delta).sum()
    mse = float(squared / gt.size)
    return dict(mae=float(absolute / gt.size), psnr=-10 * math.log10(mse) if mse else None,
                ssim=float(structural_similarity(gt, pred, data_range=1., win_size=7,
                                                use_sample_covariance=True, channel_axis=None)))


def reference_score(gt, pred, vgg=None):
    gt = np.asarray(gt, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if gt.max() == gt.min():
        return dict(mae=None, psnr=None, ssim=None)
    result = {key: float(fn(gt, pred, norm=True)) for key, fn in
              [('mae', cal_MAE), ('psnr', cal_PSNR), ('ssim', cal_SSIM)]}
    if vgg is not None:
        a, b = (torch.from_numpy(array.copy())[None,None].to('cuda:0') for array in (gt,pred))
        result['perp'] = float(cal_Perceptual(vgg, a, b, norm=True))
    return {key: value if math.isfinite(value) else None for key, value in result.items()}


def projections(volume, seg):
    return {
        'ilm_opl_mean': extract_proj_from_volume(volume, seg, 1, 'avg'),
        'opl_bm_mean': extract_proj_from_volume(volume, seg, 2, 'avg'),
        'ilm_bm_mean': extract_full_proj_from_volume(volume, seg, 'avg'),
        'full_mean': volume.mean(0, dtype=np.float32),
    }


def max_projections(volume, seg):
    return {
        'ilm_opl_max': extract_proj_from_volume(volume, seg, 1, 'max'),
        'opl_bm_max': extract_proj_from_volume(volume, seg, 2, 'max'),
        'full_max': volume.max(0),
    }


@torch.inference_mode()
def predict(model, oct_volume):
    generator = model.netG.generator_3d if CDS else model.netG
    starts = torch.linspace(0, 304-96, 4).long().tolist()
    out = torch.zeros((256,304,304), dtype=torch.float32, device='cuda:0')
    count = torch.zeros((304,304), dtype=torch.float32, device='cuda:0')
    for y in starts:
        for x in starts:
            region = np.s_[:, y:y+96, x:x+96]
            patch = torch.from_numpy(normalize(oct_volume[region]))[None,None].to('cuda:0')
            volume, _ = generator(patch)
            out[region] += volume[0,0]
            count[y:y+96, x:x+96] += 1
    out = ((out / (count + 1e-6) + 1) / 2).clamp_(0,1)
    if not torch.isfinite(out).all():
        raise FloatingPointError('Nonfinite complete OCTA volume')
    return out.cpu().numpy()


def macro(rows, group):
    names = rows[0][group]
    result = {}
    for obj, metrics in names.items():
        result[obj] = {}
        for key in metrics:
            values = [row[group][obj].get(key) for row in rows]
            result[obj][key] = float(np.mean(values)) if all(value is not None for value in values) else None
    return result


def panel(path, cid, pred, gt):
    figure = Image.new('RGB',(632,1004),'white')
    draw = ImageDraw.Draw(figure)
    draw.text((10,12),f'{cid} | GT / prediction | same display window',fill='black')
    for i,(view,label,limit) in enumerate([
            ('ilm_opl_max','ILM-OPL max',.5),('opl_bm_max','OPL-BM max',.5),('full_mean','Full-depth mean',.08)]):
        y = 44+320*i
        draw.text((10,y),f'{label}, [0,{limit}]',fill='black')
        for j,maps in enumerate((gt,pred)):
            image = Image.fromarray(np.rint(np.clip(maps[view]/limit,0,1)*255).astype(np.uint8))
            figure.paste(image,(8+312*j,y+15))
    figure.save(path)


@torch.inference_mode()
def validate(model, inputs, epoch, output=None, training_case=None):
    from models.networks_2g_st import VGGPerceptualLoss
    from common import rng_state, restore_rng

    if training_case is None:
        cases = inputs.validation
        if len(cases) != 10 or len({row['patient_id'] for row in cases}) != 10:
            raise ValueError('Full validation requires the fixed ten patients')
        phase = 'validation'
    else:
        assert training_case in inputs.train_ids
        cases = [row for row in inputs.membership if row['case_id'] == training_case]
        phase = 'training_preflight'
    folder = output or ROOT / 'reports/validation' / f'epoch_{epoch:03d}'
    folder.mkdir(parents=True, exist_ok=True)
    identity = dict(epoch=epoch, phase=phase, generator_tensor_sha256=model_digest(model.netG.state_dict()),
                    full_shape_ZYX=[256,304,304], num_windows=16)
    identity_path = folder / 'identity.json'
    if identity_path.exists() and read(identity_path) != identity:
        raise ValueError('Existing validation identity differs')
    write(identity_path, identity)
    if (folder / 'metrics.json').exists():
        return read(folder / 'metrics.json')
    rng = rng_state()
    previous = model.netG.training
    model.netG.eval()
    records = []
    try:
        vgg = model.criterionPerceptual if CDS else VGGPerceptualLoss(layers=['relu3_3'], requires_grad=False).to('cuda:0')
        for index,row in enumerate(cases,1):
            cid = row['case_id']
            record_path = folder / (cid + '.json')
            if record_path.exists():
                records.append(read(record_path))
                continue
            oct_volume, target, seg = inputs.volumes(cid)
            pred = predict(model, oct_volume)
            gt = (target / 255).astype(np.float32)
            pm, gm = projections(pred,seg), projections(gt,seg)
            pm.update(max_projections(pred,seg))
            gm.update(max_projections(gt,seg))
            standard = {'volume': score(gt,pred)}
            standard.update({name:score(gm[name].astype(np.float32),p.astype(np.float32)) for name,p in pm.items()})
            slice_scores = [score(gt[:,y,:],pred[:,y,:]) for y in range(gt.shape[1])]
            standard['bscan2d'] = {key: float(np.mean([r[key] for r in slice_scores])) if all(r[key] is not None for r in slice_scores)
                                    else None for key in ('mae','psnr','ssim')}
            q = (pred * 255).astype(np.uint8)
            reference = {'volume': reference_score(target,q)}
            qmaps, gmaps = projections(q,seg), projections(target,seg)
            reference.update({name:reference_score(gmaps[name],projection,vgg) for name,projection in qmaps.items()})
            np.savez_compressed(folder / (cid + '_projections.npz'),
                                **{'gt_'+name:a.astype(np.float32) for name,a in gm.items()},
                                **{'pred_'+name:a.astype(np.float32) for name,a in pm.items()})
            with (folder / (cid + '_bscans.csv')).open('w',encoding='utf-8',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=['y','mae','psnr','ssim'])
                writer.writeheader()
                writer.writerows(dict(y=y,**scores) for y,scores in enumerate(slice_scores))
            panel(folder / (cid+'.png'),cid,pm,gm)
            result = dict(case_id=cid,patient_id=row['patient_id'],standard=standard,reference=reference)
            write(record_path,result)
            records.append(result)
            print(f'E{epoch:03d} {phase} {index}/{len(cases)} {cid} '
                  f'shallow MIP SSIM={standard["ilm_opl_max"]["ssim"]:.5f}',flush=True)
            del pred,gt,q,oct_volume,target,seg
    finally:
        model.netG.train(previous)
        restore_rng(rng)
    summary = dict(**identity, cases=records, patient_macro={'standard':macro(records,'standard'),
                                                              'reference':macro(records,'reference')},
                   time=now(), volume_saved=False, test_used=False,
                   standard_protocol='float [0,1], fixed range1; 3D volume SSIM; 2D win7 sample covariance '
                                     'for projections and B-scans; 304 slices averaged within patient',
                   reference_protocol='XOCT uint8 prediction, GT min-max shared scaling to 255, '
                                      'masked means, author VGG19 relu3_3 Perp only on projections',
                   selection='fixed-patient macro ILM-OPL maximum projection SSIM; retain earlier on ties')
    write(folder/'metrics.json',summary)
    return summary

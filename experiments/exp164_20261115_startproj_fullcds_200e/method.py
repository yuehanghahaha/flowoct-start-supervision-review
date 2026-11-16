"""Unchanged exp018 velocity generator and exact-start loss, plus complete CDS."""
import time
import numpy as np
import torch
from torch.nn import functional as F

from fm3d.model import make_model as parent_model
from fm3d.inference import generate_volume
from cds import CDS,REGIONS,masks,project


def make_model(config):
    return parent_model(config['base_channels'],condition=True,representation='voxel')


def optimizers(model, auxiliary, config):
    g = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    settings = config['cds']['discriminator']
    d = torch.optim.Adam([{'params':auxiliary.discriminators[name].parameters()} for name in REGIONS],
                         lr=settings['lr'],betas=tuple(settings['betas']))
    return [g,d]


def batch(inputs, row, config, sampled_time):
    source,target,origin,slices = inputs.training_patch(row)
    layers = inputs.layers(row['case_id'])[:,slices[1],slices[2]] if sampled_time == 0. else None
    return dict(source=source,target=target,layers=layers,time=float(sampled_time),origin_ZYX=origin)


def gradient_norm(parameters):
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('Nonfinite discriminator gradient')
            squares.append(parameter.grad.detach().square().sum())
    norm = torch.stack(squares).sum().sqrt() if squares else torch.tensor(0.)
    if not torch.isfinite(norm) or norm <= 0:
        raise FloatingPointError('Zero/nonfinite discriminator gradient norm')
    return float(norm)


def update(model, auxiliary, opts, data, config, diagnostics=False):
    from common import projection_loss
    tick = time.perf_counter()
    device = next(model.parameters()).device
    source,target = data['source'].to(device),data['target'].to(device)
    t = torch.tensor([data['time']],dtype=torch.float32,device=device)
    fraction = t[:,None,None,None,None]
    state = (1-fraction)*source+fraction*target
    opts[0].zero_grad(set_to_none=True)
    velocity = model(state,t,source).float()
    fm = F.mse_loss(velocity,target-source)
    endpoint = state+(1-fraction)*velocity
    original_projection,valid = endpoint.new_zeros(()),0
    exact_start = data['time'] == 0.
    if exact_start:
        original_projection,_,_,valid = projection_loss(endpoint,target,data['layers'])
    active = exact_start and config['cds']['enabled'] and config['cds']['outer_weight'] != 0
    cds_loss = endpoint.new_zeros(())
    terms,dterms,dnorms,extra = {},{},{},{}
    if active:
        mask = masks(data['layers'],endpoint.shape[2],device)
        condition,prediction,truth = (project(volume,mask,config['cds']['epsilon'])
                                     for volume in (source,endpoint,target))
        auxiliary.set_d_grad(True)
        opts[1].zero_grad(set_to_none=True)
        dloss,dterms = auxiliary.discriminator_loss(condition,prediction,truth)
        if not torch.isfinite(dloss): raise FloatingPointError('Nonfinite discriminator loss')
        dloss.backward()
        dnorms = {name:gradient_norm(auxiliary.discriminators[name].parameters()) for name in REGIONS}
        opts[1].step()
        opts[1].zero_grad(set_to_none=True)
        auxiliary.set_d_grad(False)
        cds_loss,terms = auxiliary.generator_loss(condition,prediction,truth)
        if diagnostics:
            a = torch.autograd.grad(fm,velocity,retain_graph=True)[0]
            b = torch.autograd.grad(config['cds']['outer_weight']*cds_loss,velocity,retain_graph=True)[0]
            extra = dict(fm_output_gradient_rms=float(a.detach().square().mean().sqrt()),
                         cds_output_gradient_rms=float(b.detach().square().mean().sqrt()),
                         fm_cds_output_gradient_cosine=float((a*b).sum()/(a.norm()*b.norm()+1e-30)))
    loss = fm+config['lambda_projection']*original_projection+config['cds']['outer_weight']*cds_loss
    if not torch.isfinite(loss): raise FloatingPointError('Nonfinite generator loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
    if not np.isfinite(float(norm)) or norm <= 0: raise FloatingPointError('Zero/nonfinite generator gradient')
    opts[0].step()
    values = dict(loss=float(loss.detach()),fm_mse=float(fm.detach()),
                  projection_l1=float(original_projection.detach()),weighted_projection=float(config['lambda_projection']*original_projection.detach()),
                  cds_loss=float(cds_loss.detach()),weighted_cds=float(config['cds']['outer_weight']*cds_loss.detach()),
                  projection_active=float(exact_start),cds_active=float(active),d_updates=float(active),
                  exact_start=float(exact_start),time=data['time'],valid_slab_rays=float(valid),
                  gradient_norm=float(norm),forward_calls=1.,step_seconds=time.perf_counter()-tick,**extra)
    for name in REGIONS:
        values['D_'+name] = float(dterms[name].detach()) if active else 0.
        values['D_gradient_'+name] = dnorms.get(name,0.)
        for term in ('l1','gan','vgg'):
            key = name+'_'+term
            values['CDS_'+key] = float(terms[key].detach()) if active else 0.
    if not all(np.isfinite(value) for value in values.values()):
        raise FloatingPointError('Nonfinite optimization evidence')
    return values


def predict(model, source, config):
    return generate_volume(model,source,config['patch'],steps=config['ode_steps'],overlap=config['overlap'],amp=False)

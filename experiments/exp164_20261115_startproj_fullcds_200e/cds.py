"""XOCT's three regional mean-projection objectives on an FM t=0 endpoint."""
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

AUTHOR = Path(__file__).resolve().parent/'source/author'
sys.path.insert(0,str(AUTHOR))
REGIONS = ('ILM_OPL','OPL_BM','OTHER')


def masks(layers, depth, device):
    # Exact exp059 adaptation: disjoint regions in native640 MATLAB one-based coordinates.
    layers = torch.as_tensor(layers.copy(),dtype=torch.float32,device=device)
    z = ((torch.arange(depth,device=device,dtype=torch.float32)+.5)*640/depth-.5)[:,None,None]
    ilm,opl,bm = (layers[index]-1 for index in (0,2,5))
    shallow = (z >= ilm) & (z < opl)
    deep = (z >= opl) & (z <= bm)
    return torch.stack((shallow,deep,~(shallow|deep)))


def project(volume, mask, epsilon=1e-6):
    """Normalize each 3D voxel before averaging; empty rays exactly remain zero."""
    signed = 2*volume.float()-1
    count = mask.sum(1,dtype=torch.float32)
    # Input B=1,C=1; mask [region,Z,Y,X] -> regional [1,1,Y,X] projections.
    return {name:(signed*mask[i][None,None]).sum(2)/(count[i][None,None]+epsilon)
            for i,name in enumerate(REGIONS)}


class CDS(nn.Module):
    def __init__(self, config, device, load_vgg=True):
        super().__init__()
        from models import networks
        from models.networks_2g_st import GANLoss,VGGPerceptualLoss
        self.config = config
        # Preserve the FM generator's initialization and epoch crop/time streams.
        with torch.random.fork_rng(devices=[]):
            # Seed CPU initialization only; torch.manual_seed would also alter CUDA RNGs.
            torch.set_rng_state(torch.Generator(device='cpu').manual_seed(config['seed']+310000).get_state())
            self.discriminators = nn.ModuleDict({name:networks.define_D(
                input_nc=2,ndf=64,netD='basic',n_layers_D=3,norm='group',
                init_type='normal',init_gain=.02,gpu_ids=[]) for name in REGIONS})
            self.vgg = VGGPerceptualLoss(layers=['relu3_3'],requires_grad=False) if load_vgg else None
        self.gan = GANLoss(use_lsgan=True)
        self.to(device=device,dtype=torch.float32)
        if self.vgg is not None:
            self.vgg.eval().requires_grad_(False)

    def trainable_state(self):
        return self.discriminators.state_dict()

    def set_d_grad(self, value):
        self.discriminators.requires_grad_(value)

    def discriminator_loss(self, condition, prediction, target):
        terms = {}
        for name in REGIONS:
            d = self.discriminators[name]
            fake = d(torch.cat((condition[name],prediction[name].detach()),1))
            real = d(torch.cat((condition[name],target[name]),1))
            terms[name] = .5*(self.gan(fake,False)+self.gan(real,True))
        return sum(terms.values()),terms

    def generator_loss(self, condition, prediction, target):
        weights = self.config['cds']
        terms = {}
        for name in REGIONS:
            p,g = prediction[name],target[name]
            terms[name+'_l1'] = F.l1_loss(p,g)
            terms[name+'_gan'] = self.gan(self.discriminators[name](torch.cat((condition[name],p),1)),True)
            if self.vgg is None:
                raise RuntimeError('Formal CDS requires the frozen author VGG19')
            terms[name+'_vgg'] = self.vgg.perceptual_loss(p,g)
        loss = sum(weights['l1_weight']*terms[name+'_l1']+
                   weights['gan_weight']*terms[name+'_gan']+
                   weights['vgg_weight']*terms[name+'_vgg'] for name in REGIONS)
        return loss,terms

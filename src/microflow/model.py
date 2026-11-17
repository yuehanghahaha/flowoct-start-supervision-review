"""Local-first HR fusion. Mamba is a small whole-volume context provider.

Independent implementation inspired by HRNet's parallel streams and MedNeXt's
depthwise spatial mixing. No previous project model was read or copied.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ChannelNorm(nn.LayerNorm):
    def forward(self,x):return super().forward(x.movedim(1,-1)).movedim(-1,1)


class MultiScaleLocal(nn.Module):
    def __init__(self,c,t):
        super().__init__()
        self.branches=nn.ModuleList([
            nn.Conv3d(c,c,3,padding=1,groups=c),
            nn.Conv3d(c,c,(3,5,5),padding=(1,2,2),groups=c),
            nn.Conv3d(c,c,3,padding=(1,2,2),dilation=(1,2,2),groups=c)])
        self.mix=nn.Conv3d(3*c,c,1)
        self.spatial=nn.Conv3d(c,c,3,padding=1)
        self.norm=ChannelNorm(c,eps=1e-6)
        self.mod=nn.Linear(t,2*c)
        self.mlp=nn.Sequential(nn.Conv3d(c,4*c,1),nn.GELU(),nn.Conv3d(4*c,c,1))
        self.gain=nn.Parameter(torch.full((c,),.01))
        nn.init.zeros_(self.mod.weight);nn.init.zeros_(self.mod.bias)

    def forward(self,x,time):
        y=self.spatial(F.gelu(self.mix(torch.cat([f(x) for f in self.branches],1))))
        shift,scale=self.mod(time).chunk(2,1)
        y=self.norm(y)*(1+scale[:,:,None,None,None])+shift[:,:,None,None,None]
        return x+self.gain[None,:,None,None,None]*self.mlp(y)


class HRStage(nn.Module):
    def __init__(self,channels,t,depth):
        super().__init__()
        self.streams=nn.ModuleList([nn.ModuleList([MultiScaleLocal(c,t) for _ in range(depth)]) for c in channels])
        self.down=nn.ModuleList([nn.Conv3d(channels[i],channels[i+1],3,stride=2,padding=1) for i in range(2)])
        self.up=nn.ModuleList([nn.Conv3d(channels[i+1],channels[i],1) for i in range(2)])

    def forward(self,x,time):
        h=[]
        for stream,value in zip(self.streams,x):
            for block in stream:value=block(value,time)
            h.append(value)
        up0=F.interpolate(self.up[0](h[1]),size=h[0].shape[2:],mode='trilinear',align_corners=False)
        up1=F.interpolate(self.up[1](h[2]),size=h[1].shape[2:],mode='trilinear',align_corners=False)
        return [(h[0]+up0)/math.sqrt(2),
                (h[1]+self.down[0](h[0])+up1)/math.sqrt(3),
                (h[2]+self.down[1](h[1]))/math.sqrt(2)]


class AxialContext(nn.Module):
    def __init__(self,c,t,axis,factory,state,expand):
        super().__init__();self.axis=axis
        self.norm=nn.LayerNorm(c,eps=1e-6,elementwise_affine=False)
        self.scan=factory(d_model=c,d_state=state,d_conv=4,expand=expand)
        self.ffn=nn.Sequential(nn.Linear(c,2*c),nn.GELU(),nn.Linear(2*c,c))
        self.mod=nn.Linear(t,3*c)
        nn.init.zeros_(self.mod.weight);nn.init.zeros_(self.mod.bias)

    def forward(self,x,time):
        shift,scale,gate=[a[:,None,None,None] for a in self.mod(time).chunk(3,-1)]
        y=self.norm(x)*(1+scale)+shift
        order=[0]+[a for a in (1,2,3) if a!=self.axis]+[self.axis,4]
        shape=y.permute(order).shape
        lines=y.permute(order).contiguous().reshape(-1,shape[-2],shape[-1])
        y=(self.scan(lines)+self.scan(lines.flip(1)).flip(1))*.5
        y=y.reshape(shape).permute([order.index(i) for i in range(5)]).contiguous()
        return x+gate*(y+self.ffn(self.norm(y)))


class MicroVesselFlow(nn.Module):
    def __init__(self,cfg,scan_factory=None):
        super().__init__();self.cfg=cfg
        if scan_factory is None:
            from mamba_ssm import Mamba
            scan_factory=Mamba
        widths=cfg['channels'];tw=cfg['time_width'];cw=cfg['context_width']
        self.time=nn.Sequential(nn.Linear(128,tw),nn.SiLU(),nn.Linear(tw,tw))
        self.context_stem=nn.Conv3d(2,cw,3,padding=1)
        self.context_position=nn.Linear(3,cw,bias=False)
        self.context_blocks=nn.ModuleList([AxialContext(cw,tw,i%3+1,scan_factory,cfg['mamba_state'],cfg['mamba_expand'])
                                          for i in range(cfg['mamba_blocks'])])
        self.context_out=nn.Conv3d(cw,16,1)
        self.local_stem=nn.Conv3d(18,widths[0],3,padding=1)
        self.local_down=nn.ModuleList([nn.Conv3d(widths[i],widths[i+1],3,stride=2,padding=1) for i in range(2)])
        self.local_stages=nn.ModuleList([HRStage(widths,tw,cfg['blocks_per_stage']) for _ in range(cfg['stages'])])
        self.local_context=nn.ModuleList([nn.Conv3d(16,widths[0],1) for _ in range(cfg['stages'])])
        self.local_final=nn.Conv3d(sum(widths),widths[0],1)
        self.local_refine=nn.ModuleList([MultiScaleLocal(widths[0],tw) for _ in range(cfg['refine_blocks'])])
        self.local_output=nn.Conv3d(widths[0],1,1)
        nn.init.zeros_(self.local_output.weight);nn.init.zeros_(self.local_output.bias)

    def embed_time(self,t):
        freq=torch.exp(-math.log(10000)*torch.arange(64,device=t.device).float()/64)
        phase=t[:,None].float()*1000*freq
        return self.time(torch.cat([phase.cos(),phase.sin()],-1))

    def encode_context(self,oct_volume,normalized_state,t):
        grid=self.cfg['context_grid']
        x=F.adaptive_avg_pool3d(torch.cat([oct_volume,normalized_state],1),grid)
        x=self.context_stem(x).movedim(1,-1)
        coordinates=torch.stack(torch.meshgrid(*[2*(torch.arange(n,device=x.device).float()+.5)/n-1 for n in grid],indexing='ij'),-1)
        x=x+self.context_position(coordinates)[None]
        time=self.embed_time(t)
        for block in self.context_blocks:x=block(x,time)
        return self.context_out(x.movedim(-1,1))

    def forward_patch(self,oct_patch,normalized_state,t,context_patch):
        time=self.embed_time(t)
        x0=self.local_stem(torch.cat([oct_patch,normalized_state,context_patch],1))
        x1=self.local_down[0](x0);x2=self.local_down[1](x1);x=[x0,x1,x2]
        for stage,inject in zip(self.local_stages,self.local_context):
            x[0]=x[0]+.1*inject(context_patch)
            if self.training and self.cfg['checkpoint_local'] and torch.is_grad_enabled():
                x=checkpoint(stage,x,time,use_reentrant=False,preserve_rng_state=False)
            else:x=stage(x,time)
        values=[x[0]]+[F.interpolate(value,size=x[0].shape[2:],mode='trilinear',align_corners=False) for value in x[1:]]
        out=self.local_final(torch.cat(values,1))
        for block in self.local_refine:
            if self.training and self.cfg['checkpoint_local'] and torch.is_grad_enabled():
                out=checkpoint(block,out,time,use_reentrant=False,preserve_rng_state=False)
            else:out=block(out,time)
        return self.local_output(out)

    def parameter_report(self):
        context=sum(p.numel() for name,p in self.named_parameters() if name.startswith('context_'))
        total=sum(p.numel() for p in self.parameters())
        return {'total':total,'global_context':context,'local_and_time':total-context,
                'global_fraction':context/total,'local_fraction':1-context/total,
                'mamba_only':sum(p.numel() for name,p in self.named_parameters() if '.scan.' in name)}


class VesselTeacher(nn.Module):
    """Full-resolution 2D classifier: 27x27 receptive field, no spatial pooling.

It sees OCTA projections only and is frozen before flow training. Its input
derivatives remain enabled when supervising generated projections.
    """
    def __init__(self,width=32,blocks=6):
        super().__init__()
        self.stem=nn.Conv2d(1,width,3,padding=1)
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv2d(width,width,3,padding=1),
            ChannelNorm(width,eps=1e-6),nn.GELU(),nn.Conv2d(width,width,3,padding=1)) for _ in range(blocks)])
        self.head=nn.Conv2d(width,3,1)

    def forward(self,x):
        x=self.stem(x)
        for block in self.blocks:x=x+block(x)
        return self.head(F.gelu(x))

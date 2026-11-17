"""CPU-only v1 recurrence reference; actual GPU training uses official Mamba.

Algorithm/layout checked against state-spaces/mamba v2.2.4. This independently
written reference is for small-shape correctness tests and parameter counting.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ReferenceMamba(nn.Module):
    def __init__(self,d_model,d_state,d_conv,expand):
        super().__init__();inner=d_model*expand;self.rank=math.ceil(d_model/16);self.state=d_state
        self.in_proj=nn.Linear(d_model,2*inner,bias=False)
        self.conv1d=nn.Conv1d(inner,inner,d_conv,groups=inner,padding=d_conv-1)
        self.x_proj=nn.Linear(inner,self.rank+2*d_state,bias=False)
        self.dt_proj=nn.Linear(self.rank,inner)
        self.A_log=nn.Parameter(torch.arange(1,d_state+1).float().log().repeat(inner,1))
        self.D=nn.Parameter(torch.ones(inner));self.A_log._no_weight_decay=True;self.D._no_weight_decay=True
        self.out_proj=nn.Linear(inner,d_model,bias=False)
        with torch.no_grad():
            dt=torch.rand(inner)*.09+.01
            self.dt_proj.bias.copy_(dt+torch.log(-torch.expm1(-dt)))

    def forward(self,x):
        if x.is_cuda:raise RuntimeError('CPU reference may not replace the official CUDA training operator.')
        u,g=self.in_proj(x).chunk(2,-1);u=F.silu(self.conv1d(u.transpose(1,2))[...,:x.shape[1]].transpose(1,2))
        dt,b,c=self.x_proj(u).split([self.rank,self.state,self.state],-1);dt=F.softplus(self.dt_proj(dt))
        h=x.new_zeros(x.shape[0],u.shape[-1],self.state);out=[]
        a=-self.A_log.exp()
        for i in range(x.shape[1]):
            h=(dt[:,i,:,None]*a).exp()*h+dt[:,i,:,None]*u[:,i,:,None]*b[:,i,None,:]
            out.append((h*c[:,i,None,:]).sum(-1)+self.D*u[:,i])
        return self.out_proj(torch.stack(out,1)*F.silu(g))

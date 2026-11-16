"""Small-tensor reference for the v1 selective recurrence, used by checks only."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceMamba(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand):
        super().__init__()
        inner = d_model * expand
        rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.rank = rank
        self.in_proj = nn.Linear(d_model, 2 * inner, bias=False)
        self.conv1d = nn.Conv1d(inner, inner, d_conv, groups=inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(inner, rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(rank, inner)
        self.out_proj = nn.Linear(inner, d_model, bias=False)
        self.A_log = nn.Parameter(torch.arange(1, d_state + 1, dtype=torch.float32).log().repeat(inner, 1))
        self.D = nn.Parameter(torch.ones(inner))
        self.A_log._no_weight_decay = True
        self.D._no_weight_decay = True
        with torch.no_grad():
            scale = rank**-0.5
            self.dt_proj.weight.uniform_(-scale, scale)
            dt = torch.exp(torch.empty(inner).uniform_(math.log(0.001), math.log(0.1)))
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

    def forward(self, lines):
        content, gate = self.in_proj(lines).chunk(2, dim=-1)
        length = lines.shape[1]
        content = F.silu(self.conv1d(content.transpose(1, 2))[..., :length].transpose(1, 2))
        dt, b, c = self.x_proj(content).split((self.rank, self.d_state, self.d_state), dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        a = -self.A_log.exp()
        state = lines.new_zeros((lines.shape[0], a.shape[0], a.shape[1]))
        values = []
        for index in range(length):
            step = dt[:, index, :, None]
            u = content[:, index, :]
            state = (step * a).exp() * state + step * u[..., None] * b[:, index, None, :]
            values.append((state * c[:, index, None, :]).sum(dim=-1) + self.D * u)
        return self.out_proj(torch.stack(values, dim=1) * F.silu(gate))


def official_parity(device):
    from fm3d.model import official_mamba

    settings = dict(d_model=16, d_state=16, d_conv=4, expand=2)
    reference = ReferenceMamba(**settings).to(device)
    official = official_mamba(**settings).to(device)
    official.load_state_dict(reference.state_dict())
    a = torch.randn(4, 9, 16, device=device, requires_grad=True)
    b = a.detach().clone().requires_grad_(True)
    y = reference(a)
    z = official(b)
    dy, = torch.autograd.grad(y.square().mean(), a)
    dz, = torch.autograd.grad(z.square().mean(), b)
    torch.testing.assert_close(z, y, atol=2e-4, rtol=2e-3)
    torch.testing.assert_close(dz, dy, atol=2e-5, rtol=2e-3)
    return dict(max_forward_error=float((z - y).abs().max().detach()),
                max_input_gradient_error=float((dz - dy).abs().max().detach()))

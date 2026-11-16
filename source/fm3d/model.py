"""Full-grid three-scale CNN with twelve time-conditioned axial Mamba blocks.

Mamba is the unmodified official v2.2.4 operator; forward/reverse share weights.
The physical Z/Y/X axes are scanned separately, never concatenating patients.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class ChannelNorm(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.movedim(1, -1)).movedim(-1, 1)


class CNNBlock(nn.Module):
    def __init__(self, channels, time_width=None):
        super().__init__()
        self.dw = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)
        self.norm = ChannelNorm(channels, eps=1e-6)
        self.expand = nn.Conv3d(channels, 2 * channels, 1)
        self.contract = nn.Conv3d(2 * channels, channels, 1)
        self.mod = nn.Linear(time_width, 2 * channels) if time_width else None
        if self.mod is not None:
            nn.init.zeros_(self.mod.weight)
            nn.init.zeros_(self.mod.bias)

    def forward(self, x, time=None):
        y = self.norm(self.dw(x))
        if self.mod is not None:
            shift, scale = self.mod(F.silu(time)).chunk(2, dim=1)
            y = y * (1 + scale[..., None, None, None]) + shift[..., None, None, None]
        return x + self.contract(F.silu(self.expand(y)))


class AxialBlock(nn.Module):
    def __init__(self, config, axis, factory):
        super().__init__()
        c = config['global_width']
        self.axis = axis
        self.norm1 = nn.LayerNorm(c, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(c, eps=1e-6, elementwise_affine=False)
        self.scan = factory(d_model=c, d_state=config['ssm_state'], d_conv=config['ssm_conv'],
                            expand=config['ssm_expand'])
        self.mlp = nn.Sequential(nn.Linear(c, 4 * c), nn.GELU(approximate='tanh'), nn.Linear(4 * c, c))
        self.mod = nn.Linear(config['time_width'], 6 * c)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def axial(self, x):
        # x is B,Z,Y,X,C. Move only the selected physical axis into sequence position.
        order = [0] + [a for a in (1, 2, 3) if a != self.axis] + [self.axis, 4]
        reverse_order = [order.index(a) for a in range(5)]
        lines = x.permute(order).contiguous()
        shape = lines.shape
        lines = lines.reshape(-1, shape[-2], shape[-1])
        result = (self.scan(lines) + self.scan(lines.flip(1)).flip(1)) * 0.5
        return result.reshape(shape).permute(reverse_order).contiguous()

    def forward(self, x, time):
        sh1, sc1, ga1, sh2, sc2, ga2 = [v[:, None, None, None] for v in self.mod(F.silu(time)).chunk(6, -1)]
        x = x + ga1 * self.axial(self.norm1(x) * (1 + sc1) + sh1)
        return x + ga2 * self.mlp(self.norm2(x) * (1 + sc2) + sh2)


class WaveletVelocity(nn.Module):
    def __init__(self, config, scan_factory=None):
        super().__init__()
        if scan_factory is None:
            from mamba_ssm import Mamba
            scan_factory = Mamba
        self.config = config
        t, g = config['time_width'], config['global_width']
        self.time_mlp = nn.Sequential(nn.Linear(config['time_frequency_dim'], t), nn.SiLU(), nn.Linear(t, t))
        self.oct_stem = nn.Conv3d(8, 32, 3, padding=1)
        self.oct_blocks = nn.ModuleList([CNNBlock(32), CNNBlock(64), CNNBlock(96)])
        self.oct_down = nn.ModuleList([nn.Conv3d(32, 64, 2, stride=2), nn.Conv3d(64, 96, 2, stride=2)])
        self.stem = nn.Conv3d(40, 48, 1)
        self.enc0 = nn.ModuleList([CNNBlock(48, t) for _ in range(2)])
        self.down1 = nn.Conv3d(48, 96, 2, stride=2)
        self.mix1 = nn.Conv3d(160, 96, 1)
        self.enc1 = nn.ModuleList([CNNBlock(96, t) for _ in range(2)])
        self.down2 = nn.Conv3d(96, 128, 2, stride=2)
        self.mix2 = nn.Conv3d(224, 128, 1)
        self.enc2 = CNNBlock(128, t)
        self.global_in = nn.Linear(128, g)
        self.position = nn.Linear(3, g, bias=False)
        self.global_blocks = nn.ModuleList([AxialBlock(config, i % 3 + 1, scan_factory)
                                           for i in range(config['global_blocks'])])
        self.global_norm = nn.LayerNorm(g, eps=1e-6, elementwise_affine=False)
        self.global_out = nn.Linear(g, 128)
        self.dec2_mix = nn.Conv3d(256, 128, 1)
        self.dec2 = CNNBlock(128, t)
        self.up1 = nn.Conv3d(128, 96, 1)
        self.dec1_mix = nn.Conv3d(192, 96, 1)
        self.dec1 = nn.ModuleList([CNNBlock(96, t) for _ in range(2)])
        self.up0 = nn.Conv3d(96, 48, 1)
        self.dec0_mix = nn.Conv3d(96, 48, 1)
        self.dec0 = nn.ModuleList([CNNBlock(48, t) for _ in range(3)])
        self.out_norm = ChannelNorm(48, eps=1e-6)
        self.out = nn.Conv3d(48, 8, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer('band_scale', torch.ones(1, 8, 1, 1, 1))
        self.register_buffer('loss_scale', torch.ones(3))
        # Execution choices do not change the mathematical graph or parameter set.
        self.checkpoint_mode = 'selective'

    def load_statistics(self, stats):
        self.band_scale.copy_(torch.tensor(stats['band_rms']).reshape(1, 8, 1, 1, 1))
        self.loss_scale.copy_(torch.tensor([stats['volume_rms'], stats['ilm_opl_max_rms'], stats['opl_bm_max_rms']]))

    def block(self, module, *args, high_resolution=False):
        use = self.training and torch.is_grad_enabled() and (
            self.checkpoint_mode == 'all' or self.checkpoint_mode == 'selective' and
            (isinstance(module, AxialBlock) or high_resolution))
        return checkpoint(module, *args, use_reentrant=False, preserve_rng_state=False) if use else module(*args)

    def encode_oct(self, bands):
        x0 = self.block(self.oct_blocks[0], self.oct_stem(bands), high_resolution=True)
        x1 = self.block(self.oct_blocks[1], self.oct_down[0](x0))
        x2 = self.block(self.oct_blocks[2], self.oct_down[1](x1))
        return x0, x1, x2

    def embed_time(self, time):
        half = self.config['time_frequency_dim'] // 2
        frequencies = torch.exp(-math.log(10000) * torch.arange(half, device=time.device).float() / half)
        phases = time.float()[:, None] * 1000 * frequencies
        return self.time_mlp(torch.cat((phases.cos(), phases.sin()), dim=1))

    def forward(self, normalized_state, oct_features, times):
        time = self.embed_time(times)
        e0 = self.stem(torch.cat((normalized_state, oct_features[0]), dim=1))
        for block in self.enc0:
            e0 = self.block(block, e0, time, high_resolution=True)
        e1 = self.mix1(torch.cat((self.down1(e0), oct_features[1]), dim=1))
        for block in self.enc1:
            e1 = self.block(block, e1, time)
        e2 = self.block(self.enc2, self.mix2(torch.cat((self.down2(e1), oct_features[2]), dim=1)), time)
        x = self.global_in(e2.movedim(1, -1))
        coords = [2 * (torch.arange(n, device=x.device).float() + .5) / n - 1 for n in x.shape[1:4]]
        xyz = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=-1)
        x = x + self.position(xyz)[None]
        for block in self.global_blocks:
            x = self.block(block, x, time)
        x = self.global_out(self.global_norm(x)).movedim(-1, 1)
        x = self.block(self.dec2, self.dec2_mix(torch.cat((x, e2), dim=1)), time)
        x = F.interpolate(self.up1(x), size=e1.shape[2:], mode='trilinear', align_corners=False)
        x = self.dec1_mix(torch.cat((x, e1), dim=1))
        for block in self.dec1:
            x = self.block(block, x, time)
        x = F.interpolate(self.up0(x), size=e0.shape[2:], mode='trilinear', align_corners=False)
        x = self.dec0_mix(torch.cat((x, e0), dim=1))
        for block in self.dec0:
            x = self.block(block, x, time, high_resolution=True)
        return self.out(self.out_norm(x))

    def parameter_counts(self):
        return dict(total=sum(p.numel() for p in self.parameters()),
                    trainable=sum(p.numel() for p in self.parameters() if p.requires_grad),
                    global_blocks=sum(p.numel() for p in self.global_blocks.parameters()),
                    global_tokens=32 * 38 * 38, global_scan_calls=2 * len(self.global_blocks))


def official_mamba(**kwargs):
    from mamba_ssm import Mamba
    return Mamba(**kwargs)

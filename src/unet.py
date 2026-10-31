"""Conditional 2D U-Net backbone for 64x64 Pokemon generation."""

import math
import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :] * 1000.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    """Residual block with Adaptive Group Normalization (AdaGN)."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch * 2),
        )

        self.norm2 = nn.GroupNorm(32, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))

        # AdaGN: scale and shift
        scale, shift = self.emb_proj(emb).chunk(2, dim=-1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm2(h) * (1 + scale) + shift
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).transpose(1, 2)  # [B, H*W, C]
        attn_out, _ = self.attn(h, h, h)
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)
        return x + attn_out


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ConditionalUNet(nn.Module):
    """Class-conditional U-Net (~25M parameters at base_ch=128)."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_ch: int = 128,
        ch_mults: tuple[int, ...] = (1, 2, 2, 4),
        num_classes: int = 151,
        dropout: float = 0.1,
    ):
        super().__init__()
        emb_dim = base_ch * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(base_ch),
            nn.Linear(base_ch, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        # +1 for null class (unconditional fallback if needed)
        self.num_classes = num_classes
        self.class_emb = nn.Embedding(num_classes + 1, emb_dim)

        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        # Downsampling path
        self.downs = nn.ModuleList()
        ch = base_ch
        res = 64
        skip_chs = [ch]

        for i, mult in enumerate(ch_mults):
            out_ch = base_ch * mult
            for _ in range(2):
                block = nn.ModuleList([
                    ResBlock(ch, out_ch, emb_dim, dropout),
                    SelfAttentionBlock(out_ch) if res in (16, 8) else nn.Identity(),
                ])
                self.downs.append(block)
                ch = out_ch
                skip_chs.append(ch)
            if i != len(ch_mults) - 1:
                self.downs.append(nn.ModuleList([Downsample(ch)]))
                res //= 2
                skip_chs.append(ch)

        # Bottleneck
        self.mid_block1 = ResBlock(ch, ch, emb_dim, dropout)
        self.mid_attn = SelfAttentionBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, emb_dim, dropout)

        # Upsampling path
        self.ups = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mults))):
            out_ch = base_ch * mult
            for _ in range(3):
                skip_ch = skip_chs.pop()
                block = nn.ModuleList([
                    ResBlock(ch + skip_ch, out_ch, emb_dim, dropout),
                    SelfAttentionBlock(out_ch) if res in (16, 8) else nn.Identity(),
                ])
                self.ups.append(block)
                ch = out_ch
            if i != 0:
                self.ups.append(nn.ModuleList([Upsample(ch)]))
                res *= 2

        self.final_norm = nn.GroupNorm(32, ch)
        self.final_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        # Zero-init final conv for stable initial flow matching training
        nn.init.zeros_(self.final_conv.weight)
        nn.init.zeros_(self.final_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        B = x.shape[0]
        if timestep.dim() == 0:
            timestep = timestep.expand(B)
        elif timestep.dim() > 1:
            timestep = timestep.view(B)

        emb = self.time_mlp(timestep.float())

        if category is None:
            category = torch.full((B,), self.num_classes, device=x.device, dtype=torch.long)
        emb = emb + self.class_emb(category)

        h = self.init_conv(x)
        skips = [h]

        for layer in self.downs:
            if len(layer) == 1:  # Downsample
                h = layer[0](h)
                skips.append(h)
            else:
                res_block, attn_block = layer
                h = res_block(h, emb)
                h = attn_block(h)
                skips.append(h)

        h = self.mid_block1(h, emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, emb)

        for layer in self.ups:
            if len(layer) == 1:  # Upsample
                h = layer[0](h)
            else:
                res_block, attn_block = layer
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = res_block(h, emb)
                h = attn_block(h)

        return self.final_conv(F.silu(self.final_norm(h)))
import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from einops import rearrange

from xdiffusion.layers.flux import Modulation, SelfAttention, attention, apply_rope


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False
        )

    def forward(self, x):
        return self.pool(x) - x


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        pool_size: int = 3,
        qkv_bias: bool = False,
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = torch.nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )

        self.img_norm2 = torch.nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.img_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.img_proj = nn.Linear(hidden_size, hidden_size)

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = torch.nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.txt_norm2 = torch.nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.txt_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            torch.nn.GELU(approximate="tanh"),
            torch.nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.txt_proj = nn.Linear(hidden_size, hidden_size)

        self.token_mixer = Pooling(pool_size=pool_size)

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        attn_mask: Tensor = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of block.

        Args:
            img: Tensor batch of noised image (B, L, D)
            txt: Tensor batch of text embeddings (B, L, D)
            vec: Tensor batch of timestep (and other) embeddings (B, L, D)
            pe: Tensor batch of positional encodings (B, L, D)
        """
        # Separate moduation for text and image pathways
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift

        # Start the token mixing (replacement for attention).
        # First apply the positional embedding. Tokens are currently shape (B, L, D),
        # But rope assumes a multi-head "attention", so insert a singlular head dimension.
        txt_modulated = rearrange(
            txt_modulated, "B L (H D) -> B H L D", H=self.num_heads
        )
        img_modulated = rearrange(
            img_modulated, "B L (H D) -> B H L D", H=self.num_heads
        )

        modulated = torch.cat((txt_modulated, img_modulated), dim=2)
        modulated, _ = apply_rope(modulated, modulated, pe)

        # Now mix the tokens.
        attn = self.token_mixer(modulated)
        attn = rearrange(attn, "B H L D -> B L (H D)")

        # Mask out any tokens that should not take part in token mixing.
        # The token mask is a boolean tensor of shape (L,L) where
        # a value of True indicates the token should take part in attention.
        assert attn_mask is None

        # Pull out the text and image tokens again
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img bloks. Note this is NOT using the parallel
        # attention layers from https://arxiv.org/abs/2302.05442, but rather
        # the standard MMDiT calculation from https://arxiv.org/abs/2403.03206.
        img = img + img_mod1.gate * self.img_proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt bloks. Note this is NOT using the parallel
        # attention layers from https://arxiv.org/abs/2302.05442, but rather
        # the standard MMDiT calculation from https://arxiv.org/abs/2403.03206.
        txt = txt + txt_mod1.gate * self.txt_proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )
        return img, txt


class TripleStreamBlock(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False
    ):
        super().__init__()

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Spatial processing
        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # Temporal processing
        self.temporal_mod = Modulation(hidden_size, double=True)
        self.temporal_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.temporal_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )
        self.temporal_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.temporal_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # Textual processing
        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(
            dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias
        )

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(
        self, img: Tensor, temporal: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Forward pass of block.

        Args:
            img: Tensor batch of noised spatial image data (B, L, D)
            temporal: Tensor batch of noised temporal data (B, L', D)
            txt: Tensor batch of text embeddings (B, L, D)
            vec: Tensor batch of timestep (and other) embeddings (B, L, D)
            pe: Tensor batch of positional encodings (B, L, D)
        """
        # Separate moduation for text and image pathways
        img_mod1, img_mod2 = self.img_mod(vec)
        temporal_mod1, temporal_mod2 = self.temporal_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        # QK normalization, per https://arxiv.org/abs/2302.05442
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # Prepare spatial for attention
        temporal_modulated = self.temporal_norm1(temporal)
        temporal_modulated = (
            1 + temporal_mod1.scale
        ) * temporal_modulated + temporal_mod1.shift
        temporal_qkv = self.temporal_attn.qkv(temporal_modulated)
        temporal_q, temporal_k, temporal_v = rearrange(
            temporal_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        # QK normalization, per https://arxiv.org/abs/2302.05442
        temporal_q, temporal_k = self.temporal_attn.norm(
            temporal_q, temporal_k, temporal_v
        )

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        # QK normalization, per https://arxiv.org/abs/2302.05442
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q, temporal_q), dim=2)
        k = torch.cat((txt_k, img_k, temporal_k), dim=2)
        v = torch.cat((txt_v, img_v, temporal_q), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn, temporal_attn = (
            attn[:, : txt.shape[1]],
            attn[:, txt.shape[1] : txt.shape[1] + img.shape[1]],
            attn[:, txt.shape[1] + img.shape[1] :],
        )

        # calculate the img bloks. Note this is NOT using the parallel
        # attention layers from https://arxiv.org/abs/2302.05442, but rather
        # the standard MMDiT calculation from https://arxiv.org/abs/2403.03206.
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the img bloks. Note this is NOT using the parallel
        # attention layers from https://arxiv.org/abs/2302.05442, but rather
        # the standard MMDiT calculation from https://arxiv.org/abs/2403.03206.
        temporal = temporal + temporal_mod1.gate * self.temporal_attn.proj(
            temporal_attn
        )
        temporal = temporal + temporal_mod2.gate * self.temporal_mlp(
            (1 + temporal_mod2.scale) * self.temporal_norm2(img) + temporal_mod2.shift
        )

        # calculate the txt bloks. Note this is NOT using the parallel
        # attention layers from https://arxiv.org/abs/2302.05442, but rather
        # the standard MMDiT calculation from https://arxiv.org/abs/2403.03206.
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )
        return img, temporal, txt

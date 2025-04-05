import math
import torch
import torch.nn as nn
from typing import Dict, List

from xdiffusion.layers.hunyuan_video.rope import get_rotary_pos_embed
from xdiffusion.layers.modulate import modulate
from xdiffusion.layers.utils import to_2tuple

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding

    Image to Patch Embedding using Conv2d

    A convolution based approach to patchifying a 2D image w/ embedding projection.

    Based on the impl in https://github.com/google-research/vision_transformer

    Hacked together by / Copyright 2020 Ross Wightman

    Remove the _assert function in forward function to be compatible with multi-resolution images.
    """

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        if bias:
            nn.init.zeros_(self.proj.bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TextProjection(nn.Module):
    """
    Projects text embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_channels, hidden_size, act_layer):
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=in_channels,
            out_features=hidden_size,
            bias=True,
        )
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        act_layer,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
    ):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size, hidden_size, bias=True
            ),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(
            t, self.frequency_embedding_size, self.max_period
        ).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class FinalLayer(nn.Module):
    """The final layer of DiT.

    Supports non-square patch sizes.
    """

    def __init__(
        self, hidden_size, patch_size, out_channels, act_layer
    ):
        super().__init__()

        # Just use LayerNorm for the final layer
        self.norm_final = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6
        )
        if isinstance(patch_size, int):
            self.linear = nn.Linear(
                hidden_size,
                patch_size * patch_size * out_channels,
                bias=True,
            )
        else:
            self.linear = nn.Linear(
                hidden_size,
                patch_size[0] * patch_size[1] * patch_size[2] * out_channels,
                bias=True,
            )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Here we don't distinguish between the modulate types. Just use the simple one.
        self.adaLN_modulation = nn.Sequential(
            act_layer(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x

class RopeFrequencies(nn.Module):
    def __init__(
        self,
        context_output_key: str,
        video_length: int,
        height: int,
        width: int,
        patch_size: int,
        rope_theta: float,
        model_hidden_size: int,
        model_heads_num: int,
        rope_dim_list: List[int],
        vae_spec: str,
        **kwargs,
    ):
        super().__init__()
        self.context_output_key = context_output_key
        self.video_length = video_length
        self.video_height = height
        self.video_width = width
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.model_hidden_size = model_hidden_size
        self.model_heads_num = model_heads_num
        self.rope_dim_list = rope_dim_list
        self.vae_spec = vae_spec

    def forward(
        self,
        context: Dict,
        device,
        **kwargs,
    ):
        freqs_cos, freqs_sin = get_rotary_pos_embed(
            video_length=self.video_length,
            height=self.video_height,
            width=self.video_width,
            patch_size=self.patch_size,
            rope_theta=self.rope_theta,
            model_hidden_size=self.model_hidden_size,
            model_heads_num=self.model_heads_num,
            rope_dim_list=self.rope_dim_list,
            vae_spec=self.vae_spec
        )
        context[self.context_output_key + "_cos"] = freqs_cos.to(device)
        context[self.context_output_key + "_sin"] = freqs_sin.to(device)
        return context

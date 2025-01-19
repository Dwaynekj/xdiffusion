"""LTX Video 3D Transformer Score Network

Based on the Pixart-Alpha transformer, updated for 3D and utilizing
QK Norm, ROPE positional embedding, and RMS Norm instead of LayerNorm.
"""

import math
from dataclasses import dataclass
from einops import rearrange
from typing import Any, Dict, List, Optional, Literal, Tuple, Union
import os
import json
import glob
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open

from xdiffusion.layers.embedding import PixArtAlphaTextProjection
from xdiffusion.layers.ltx import BasicTransformerBlock, SkipLayerStrategy
from xdiffusion.layers.norm import AdaLayerNormSingle
from xdiffusion.layers.utils import get_3d_sincos_pos_embed
from xdiffusion.utils import DotConfig

ORIGINAL_TRANSFORMER_CONFIG = {
    "activation_fn": "gelu-approximate",
    "attention_bias": True,
    "attention_head_dim": 64,
    "attention_type": "default",
    "caption_channels": 4096,
    "cross_attention_dim": 2048,
    "double_self_attention": False,
    "dropout": 0.0,
    "in_channels": 128,
    "norm_elementwise_affine": False,
    "norm_eps": 1e-06,
    "norm_num_groups": 32,
    "num_attention_heads": 32,
    "num_embeds_ada_norm": 1000,
    "num_layers": 28,
    "num_vector_embeds": None,
    "only_cross_attention": False,
    "out_channels": 128,
    "project_to_2d_pos": True,
    "upcast_attention": False,
    "use_linear_projection": False,
    "qk_norm": "rms_norm",
    "standardization_norm": "rms_norm",
    "positional_embedding_type": "rope",
    "positional_embedding_theta": 10000.0,
    "positional_embedding_max_pos": [20, 2048, 2048],
    "timestep_scale_multiplier": 1000,
}


class LTXVideoTransformer(torch.nn.Module):

    def __init__(
        self,
        config: DotConfig,
    ):
        super().__init__()

        self._config = config
        self.use_linear_projection = config.use_linear_projection
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_dim = config.attention_head_dim
        inner_dim = config.num_attention_heads * config.attention_head_dim
        self.inner_dim = inner_dim

        self.project_to_2d_pos = config.project_to_2d_pos

        self.patchify_proj = nn.Linear(config.input_channels, inner_dim, bias=True)

        self.positional_embedding_type = config.positional_embedding_type
        self.positional_embedding_theta = config.positional_embedding_theta
        self.positional_embedding_max_pos = config.positional_embedding_max_pos
        self.use_rope = self.positional_embedding_type == "rope"
        self.timestep_scale_multiplier = config.timestep_scale_multiplier

        if self.positional_embedding_type == "absolute":
            embed_dim_3d = (
                math.ceil((inner_dim / 2) * 3)
                if config.project_to_2d_pos
                else inner_dim
            )
            if self.project_to_2d_pos:
                self.to_2d_proj = torch.nn.Linear(embed_dim_3d, inner_dim, bias=False)
                self._init_to_2d_proj_weights(self.to_2d_proj)
        elif self.positional_embedding_type == "rope":
            if config.positional_embedding_theta is None:
                raise ValueError(
                    "If `positional_embedding_type` type is rope, `positional_embedding_theta` must also be defined"
                )
            if config.positional_embedding_max_pos is None:
                raise ValueError(
                    "If `positional_embedding_type` type is rope, `positional_embedding_max_pos` must also be defined"
                )

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    config.num_attention_heads,
                    config.attention_head_dim,
                    dropout=config.dropout,
                    cross_attention_dim=config.cross_attention_dim,
                    activation_fn=config.activation_fn,
                    num_embeds_ada_norm=config.num_embeds_ada_norm,
                    attention_bias=config.attention_bias,
                    only_cross_attention=config.only_cross_attention,
                    double_self_attention=config.double_self_attention,
                    upcast_attention=config.upcast_attention,
                    adaptive_norm=config.adaptive_norm,
                    standardization_norm=config.standardization_norm,
                    norm_elementwise_affine=config.norm_elementwise_affine,
                    norm_eps=config.norm_eps,
                    attention_type=config.attention_type,
                    qk_norm=config.qk_norm,
                    use_rope=self.use_rope,
                )
                for d in range(config.num_layers)
            ]
        )

        # 4. Define output layers
        self.out_channels = (
            config.input_channels
            if config.out_channels is None
            else config.out_channels
        )
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, inner_dim) / inner_dim**0.5
        )
        self.proj_out = nn.Linear(inner_dim, self.out_channels)

        self.adaln_single = AdaLayerNormSingle(
            inner_dim, use_additional_conditions=False
        )
        if config.adaptive_norm == "single_scale":
            self.adaln_single.linear = nn.Linear(inner_dim, 4 * inner_dim, bias=True)

        self.caption_projection = None
        if config.caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=config.caption_channels, hidden_size=inner_dim
            )

        self.gradient_checkpointing = False

    def create_skip_layer_mask(
        self,
        skip_block_list: List[int],
        batch_size: int,
        num_conds: int,
        ptb_index: int,
    ):
        num_layers = len(self.transformer_blocks)
        mask = torch.ones(
            (num_layers, batch_size * num_conds), device=self.device, dtype=self.dtype
        )
        for block_idx in skip_block_list:
            mask[block_idx, ptb_index::num_conds] = 0
        return mask

    def initialize(self, embedding_std: float, mode: Literal["ltx_video", "legacy"]):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(
            self.adaln_single.emb.timestep_embedder.linear_1.weight, std=embedding_std
        )
        nn.init.normal_(
            self.adaln_single.emb.timestep_embedder.linear_2.weight, std=embedding_std
        )
        nn.init.normal_(self.adaln_single.linear.weight, std=embedding_std)

        if hasattr(self.adaln_single.emb, "resolution_embedder"):
            nn.init.normal_(
                self.adaln_single.emb.resolution_embedder.linear_1.weight,
                std=embedding_std,
            )
            nn.init.normal_(
                self.adaln_single.emb.resolution_embedder.linear_2.weight,
                std=embedding_std,
            )
        if hasattr(self.adaln_single.emb, "aspect_ratio_embedder"):
            nn.init.normal_(
                self.adaln_single.emb.aspect_ratio_embedder.linear_1.weight,
                std=embedding_std,
            )
            nn.init.normal_(
                self.adaln_single.emb.aspect_ratio_embedder.linear_2.weight,
                std=embedding_std,
            )

        # Initialize caption embedding MLP:
        nn.init.normal_(self.caption_projection.linear_1.weight, std=embedding_std)
        nn.init.normal_(self.caption_projection.linear_1.weight, std=embedding_std)

        for block in self.transformer_blocks:
            if mode.lower() == "ltx_video":
                nn.init.constant_(block.attn1.to_out[0].weight, 0)
                nn.init.constant_(block.attn1.to_out[0].bias, 0)

            nn.init.constant_(block.attn2.to_out[0].weight, 0)
            nn.init.constant_(block.attn2.to_out[0].bias, 0)

            if mode.lower() == "ltx_video":
                nn.init.constant_(block.ff.net[2].weight, 0)
                nn.init.constant_(block.ff.net[2].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    @staticmethod
    def _init_to_2d_proj_weights(linear_layer):
        input_features = linear_layer.weight.data.size(1)
        output_features = linear_layer.weight.data.size(0)

        # Start with a zero matrix
        identity_like = torch.zeros((output_features, input_features))

        # Fill the diagonal with 1's as much as possible
        min_features = min(output_features, input_features)
        identity_like[:min_features, :min_features] = torch.eye(min_features)
        linear_layer.weight.data = identity_like.to(linear_layer.weight.data.device)

    def get_fractional_positions(self, indices_grid):
        fractional_positions = torch.stack(
            [
                indices_grid[:, i] / self.positional_embedding_max_pos[i]
                for i in range(3)
            ],
            dim=-1,
        )
        return fractional_positions

    def precompute_freqs_cis(self, indices_grid, output_dtype, spacing="exp"):
        dtype = torch.float32  # We need full precision in the freqs_cis computation.
        dim = self.inner_dim
        theta = self.positional_embedding_theta

        fractional_positions = self.get_fractional_positions(indices_grid)

        start = 1
        end = theta
        device = fractional_positions.device
        if spacing == "exp":
            indices = theta ** (
                torch.linspace(
                    math.log(start, theta),
                    math.log(end, theta),
                    dim // 6,
                    device=device,
                    dtype=dtype,
                )
            )
            indices = indices.to(dtype=dtype)
        elif spacing == "exp_2":
            indices = 1.0 / theta ** (torch.arange(0, dim, 6, device=device) / dim)
            indices = indices.to(dtype=dtype)
        elif spacing == "linear":
            indices = torch.linspace(start, end, dim // 6, device=device, dtype=dtype)
        elif spacing == "sqrt":
            indices = torch.linspace(
                start**2, end**2, dim // 6, device=device, dtype=dtype
            ).sqrt()

        indices = indices * math.pi / 2

        if spacing == "exp_2":
            freqs = (
                (indices * fractional_positions.unsqueeze(-1))
                .transpose(-1, -2)
                .flatten(2)
            )
        else:
            freqs = (
                (indices * (fractional_positions.unsqueeze(-1) * 2 - 1))
                .transpose(-1, -2)
                .flatten(2)
            )

        cos_freq = freqs.cos().repeat_interleave(2, dim=-1)
        sin_freq = freqs.sin().repeat_interleave(2, dim=-1)
        if dim % 6 != 0:
            cos_padding = torch.ones_like(cos_freq[:, :, : dim % 6])
            sin_padding = torch.zeros_like(cos_freq[:, :, : dim % 6])
            cos_freq = torch.cat([cos_padding, cos_freq], dim=-1)
            sin_freq = torch.cat([sin_padding, sin_freq], dim=-1)
        return cos_freq.to(output_dtype), sin_freq.to(output_dtype)

    def forward(
        self,
        x: torch.Tensor,
        context: Dict,
        **kwargs
        # indices_grid: torch.Tensor,
        # encoder_hidden_states: Optional[torch.Tensor] = None,
        # timestep: Optional[torch.LongTensor] = None,
        # class_labels: Optional[torch.LongTensor] = None,
        # cross_attention_kwargs: Dict[str, Any] = None,
        # attention_mask: Optional[torch.Tensor] = None,
        # encoder_attention_mask: Optional[torch.Tensor] = None,
        # skip_layer_mask: Optional[torch.Tensor] = None,
        # skip_layer_strategy: Optional[SkipLayerStrategy] = None,
        # return_dict: bool = True,
    ):
        B, C, F, H, W = x.shape

        timestep = context["timestep"]
        encoder_hidden_states = context["text_embeddings"]
        encoder_attention_mask = context["text_attention_mask"]
        attention_mask = None
        skip_layer_mask = None
        skip_layer_strategy = SkipLayerStrategy.Attention
        cross_attention_kwargs = {}
        class_labels = None

        indices_grid = get_grid(
            num_frames=F,
            height=H,
            width=W,
            batch_size=B,
            scale_grid=None,
            device=x.device,
        )

        # Patchify the input by rearranging to (B, F*H*W, C)
        hidden_states = rearrange(x, "b c f h w -> b c (f h w)").transpose(-1, -2)

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        hidden_states = self.patchify_proj(hidden_states)

        if self.timestep_scale_multiplier:
            timestep = self.timestep_scale_multiplier * timestep

        if self.positional_embedding_type == "absolute":
            pos_embed_3d = self.get_absolute_pos_embed(indices_grid).to(
                hidden_states.device
            )
            if self.project_to_2d_pos:
                pos_embed = self.to_2d_proj(pos_embed_3d)
            hidden_states = (hidden_states + pos_embed).to(hidden_states.dtype)
            freqs_cis = None
        elif self.positional_embedding_type == "rope":
            freqs_cis = self.precompute_freqs_cis(indices_grid, hidden_states.dtype)

        batch_size = hidden_states.shape[0]
        timestep, embedded_timestep = self.adaln_single(
            timestep.flatten(),
            {"resolution": None, "aspect_ratio": None},
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )
        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.shape[-1]
        )

        if skip_layer_mask is None:
            skip_layer_mask = torch.ones(
                len(self.transformer_blocks), batch_size, device=hidden_states.device
            )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, -1, hidden_states.shape[-1]
            )

        for block_idx, block in enumerate(self.transformer_blocks):
            hidden_states = block(
                hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
                skip_layer_mask=skip_layer_mask[block_idx],
                skip_layer_strategy=skip_layer_strategy,
            )

        # 3. Output
        scale_shift_values = (
            self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        hidden_states = self.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        hidden_states = rearrange(
            hidden_states.transpose(-1, -2), "b c (f h w) -> b c f h w", h=H, w=W, f=F
        )

        return hidden_states

    def get_absolute_pos_embed(self, grid):
        grid_np = grid[0].cpu().numpy()
        embed_dim_3d = (
            math.ceil((self.inner_dim / 2) * 3)
            if self.project_to_2d_pos
            else self.inner_dim
        )
        pos_embed = get_3d_sincos_pos_embed(  # (f h w)
            embed_dim_3d,
            grid_np,
            h=int(max(grid_np[1]) + 1),
            w=int(max(grid_np[2]) + 1),
            f=int(max(grid_np[0] + 1)),
        )
        return torch.from_numpy(pos_embed).float().unsqueeze(0)


def get_grid(num_frames, height, width, batch_size, scale_grid, device):
    f = num_frames
    h = height
    w = width
    grid_h = torch.arange(h, dtype=torch.float32, device=device)
    grid_w = torch.arange(w, dtype=torch.float32, device=device)
    grid_f = torch.arange(f, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_f, grid_h, grid_w)
    grid = torch.stack(grid, dim=0)
    grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

    if scale_grid is not None:
        for i in range(3):
            if isinstance(scale_grid[i], Tensor):
                scale = append_dims(scale_grid[i], grid.ndim - 1)
            else:
                scale = scale_grid[i]
            grid[:, i, ...] = grid[:, i, ...] * scale * self._patch_size[i]

    grid = rearrange(grid, "b c f h w -> b c (f h w)", b=batch_size)
    return grid

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from xdiffusion.autoencoders.opensora.hunyuan.unet_causal_3d_blocks import (
    CausalConv3d,
    DownEncoderBlockCausal3D,
    UNetMidBlockCausal3D,
    UpDecoderBlockCausal3D,
    prepare_causal_attention_mask,
)
from xdiffusion.autoencoders.distributions import BaseOutput


@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.FloatTensor


class EncoderCausal3D(nn.Module):
    r"""
    The `EncoderCausal3D` layer of a variational autoencoder that encodes its input into a latent representation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1
        )
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, _ in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_downsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_downsample_layers = int(np.log2(time_compression_ratio))
            if time_compression_ratio == 4:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(
                    i >= (len(block_out_channels) - 1 - num_time_downsample_layers)
                    and not is_final_block
                )
            elif time_compression_ratio == 8:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(i < num_spatial_downsample_layers)
            elif time_compression_ratio == 2:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(i < num_time_downsample_layers)
            else:
                raise ValueError(
                    f"Unsupported time_compression_ratio: {time_compression_ratio}."
                )

            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (2,) if add_time_downsample else (1,)
            downsample_stride = tuple(downsample_stride_T + downsample_stride_HW)
            down_block = DownEncoderBlockCausal3D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                dropout=dropout,
                add_downsample=bool(add_spatial_downsample or add_time_downsample),
                downsample_stride=downsample_stride,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )

            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = CausalConv3d(
            block_out_channels[-1], conv_out_channels, kernel_size=3
        )

    def prepare_attention_mask(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = hidden_states.shape
        attention_mask = prepare_causal_attention_mask(
            T, H * W, hidden_states.dtype, hidden_states.device, batch_size=B
        )
        return attention_mask

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `EncoderCausal3D` class."""
        assert len(sample.shape) == 5, "The input tensor should have 5 dimensions"

        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle
        if self.mid_block.add_attention:
            attention_mask = self.prepare_attention_mask(sample)
        else:
            attention_mask = None
        # sample = auto_grad_checkpoint(self.mid_block, sample, attention_mask)
        sample = self.mid_block(sample, attention_mask)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class DecoderCausal3D(nn.Module):
    r"""
    The `DecoderCausal3D` layer of a variational autoencoder that decodes its latent representation into an output sample.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention=True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(
            in_channels, block_out_channels[-1], kernel_size=3, stride=1
        )
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, _ in enumerate(block_out_channels):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_upsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_upsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial_upsample = bool(i < num_spatial_upsample_layers)
                add_time_upsample = bool(
                    i >= len(block_out_channels) - 1 - num_time_upsample_layers
                    and not is_final_block
                )
            elif time_compression_ratio == 8:
                add_spatial_upsample = bool(i < num_spatial_upsample_layers)
                add_time_upsample = bool(i < num_spatial_upsample_layers)
            elif time_compression_ratio == 2:
                add_spatial_upsample = bool(i < num_spatial_upsample_layers)
                add_time_upsample = bool(i < num_time_upsample_layers)
            else:
                raise ValueError(
                    f"Unsupported time_compression_ratio: {time_compression_ratio}."
                )

            upsample_scale_factor_HW = (2, 2) if add_spatial_upsample else (1, 1)
            upsample_scale_factor_T = (2,) if add_time_upsample else (1,)
            upsample_scale_factor = tuple(
                upsample_scale_factor_T + upsample_scale_factor_HW
            )
            up_block = UpDecoderBlockCausal3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                resolution_idx=None,
                dropout=dropout,
                add_upsample=bool(add_spatial_upsample or add_time_upsample),
                upsample_scale_factor=upsample_scale_factor,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(block_out_channels[0], out_channels, kernel_size=3)

    def post_process(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        return sample

    def prepare_attention_mask(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = hidden_states.shape
        attention_mask = prepare_causal_attention_mask(
            T, H * W, hidden_states.dtype, hidden_states.device, batch_size=B
        )
        return attention_mask

    def forward(
        self,
        sample: torch.FloatTensor,
    ) -> torch.FloatTensor:
        r"""The forward method of the `DecoderCausal3D` class."""
        assert len(sample.shape) == 5, "The input tensor should have 5 dimensions."

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        # middle
        if self.mid_block.add_attention:
            attention_mask = self.prepare_attention_mask(sample)
        else:
            attention_mask = None

        # sample = auto_grad_checkpoint(self.mid_block, sample, attention_mask)
        sample = self.mid_block(sample, attention_mask)
        sample = sample.to(upscale_dtype)

        # up
        for up_block in self.up_blocks:
            sample = up_block(sample)

        # post-process
        # if getattr(self, "grad_checkpointing", False):
        #     sample = checkpoint(self.post_process, sample, use_reentrant=True)
        # else:
        sample = self.post_process(sample)

        sample = self.conv_out(sample)

        return sample

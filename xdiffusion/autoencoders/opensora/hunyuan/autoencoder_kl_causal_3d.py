from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

from xdiffusion.layers.attention_diffusers import Attention
from xdiffusion.autoencoders.opensora.hunyuan.vae import (
    DecoderCausal3D,
    EncoderCausal3D,
    DecoderOutput,
)

from xdiffusion.autoencoders.distributions import (
    DiagonalGaussianDistribution,
)
from xdiffusion.utils import (
    DotConfig,
    instantiate_from_config,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)


@dataclass
class AutoEncoder3DConfig:
    from_pretrained: str | None
    act_fn: str = "silu"
    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scale_factor: float = 0.476986
    shift_factor: float = 0
    time_compression_ratio: int = 4
    spatial_compression_ratio: int = 8
    mid_block_add_attention: bool = True
    block_out_channels: tuple[int] = (128, 256, 512, 512)
    sample_size: int = 256
    sample_tsize: int = 64
    use_slicing: bool = False
    use_spatial_tiling: bool = False
    use_temporal_tiling: bool = False
    tile_overlap_factor: float = 0.25
    dropout: float = 0.0
    channel: bool = False


class AutoencoderKLCausal3D(torch.nn.Module):
    r"""
    A VAE model with KL loss for encoding images/videos into latents and decoding latent representations into images/videos.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    def __init__(self, config: DotConfig):
        super().__init__()
        self.config = config
        self.scale_factor = config.scale_factor
        self.shift_factor = config.shift_factor

        self.time_compression_ratio = config.time_compression_ratio
        self.spatial_compression_ratio = config.spatial_compression_ratio
        self.z_channels = config.latent_channels

        self.encoder = EncoderCausal3D(
            in_channels=config.in_channels,
            out_channels=config.latent_channels,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            act_fn=config.act_fn,
            norm_num_groups=config.norm_num_groups,
            double_z=True,
            time_compression_ratio=config.time_compression_ratio,
            spatial_compression_ratio=config.spatial_compression_ratio,
            mid_block_add_attention=config.mid_block_add_attention,
            dropout=config.dropout,
        )

        self.decoder = DecoderCausal3D(
            in_channels=config.latent_channels,
            out_channels=config.out_channels,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            norm_num_groups=config.norm_num_groups,
            act_fn=config.act_fn,
            time_compression_ratio=config.time_compression_ratio,
            spatial_compression_ratio=config.spatial_compression_ratio,
            mid_block_add_attention=config.mid_block_add_attention,
            dropout=config.dropout,
        )

        self.quant_conv = nn.Conv3d(
            2 * config.latent_channels, 2 * config.latent_channels, kernel_size=1
        )
        self.post_quant_conv = nn.Conv3d(
            config.latent_channels, config.latent_channels, kernel_size=1
        )

        self.use_slicing = config.use_slicing
        self.use_spatial_tiling = config.use_spatial_tiling
        self.use_temporal_tiling = config.use_temporal_tiling

        # only relevant if vae tiling is enabled
        self.tile_sample_min_tsize = config.sample_tsize
        self.tile_latent_min_tsize = (
            config.sample_tsize // config.time_compression_ratio
        )

        self.tile_sample_min_size = config.sample_size
        sample_size = (
            config.sample_size[0]
            if isinstance(config.sample_size, (list, tuple))
            else config.sample_size
        )
        self.tile_latent_min_size = int(
            sample_size / (2 ** (len(config.block_out_channels) - 1))
        )
        self.tile_overlap_factor = config.tile_overlap_factor
        self.loss = instantiate_from_config(config.loss_config._cfg)

    def encode(
        self,
        x: torch.FloatTensor,
        sample_posterior: bool = True,
        return_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> Union[torch.FloatTensor, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images/videos into latents.

        Args:
            x (`torch.FloatTensor`): Input batch of images/videos.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
                The latent representations of the encoded images/videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        assert len(x.shape) == 5, "The input tensor should have 5 dimensions."

        x = normalize_to_neg_one_to_one(x)

        if self.use_temporal_tiling and x.shape[2] > self.tile_sample_min_tsize:
            posterior = self.temporal_tiled_encode(x)
        elif self.use_spatial_tiling and (
            x.shape[-1] > self.tile_sample_min_size
            or x.shape[-2] > self.tile_sample_min_size
        ):
            posterior = self.spatial_tiled_encode(x)
        else:
            if self.use_slicing and x.shape[0] > 1:
                encoded_slices = [self.encoder(x_slice) for x_slice in x.split(1)]
                h = torch.cat(encoded_slices)
            else:
                h = self.encoder(x)
            moments = self.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)

        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()

        z = self.scale_factor * (z - self.shift_factor)  # shift & scale

        if return_posterior:
            return z, posterior
        else:
            return z

    def _decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        assert len(z.shape) == 5, "The input tensor should have 5 dimensions."

        if self.use_temporal_tiling and z.shape[2] > self.tile_latent_min_tsize:
            return self.temporal_tiled_decode(z, return_dict=return_dict)

        if self.use_spatial_tiling and (
            z.shape[-1] > self.tile_latent_min_size
            or z.shape[-2] > self.tile_latent_min_size
        ):
            return self.spatial_tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def decode(self, z: torch.FloatTensor) -> torch.FloatTensor:
        """
        Decode a batch of images/videos.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        z = z / self.scale_factor + self.shift_factor  # scale & shift

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample
        return unnormalize_to_zero_to_one(decoded)

    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (
                1 - y / blend_extent
            ) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (
                1 - x / blend_extent
            ) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def blend_t(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (
                1 - x / blend_extent
            ) + b[:, :, x, :, :] * (x / blend_extent)
        return b

    def spatial_tiled_encode(
        self, x: torch.FloatTensor, return_moments: bool = False
    ) -> DiagonalGaussianDistribution:
        r"""Encode a batch of images/videos using a tiled encoder.

        When this option is enabled, the VAE will split the input tensor into tiles to compute encoding in several
        steps. This is useful to keep memory use constant regardless of image/videos size. The end result of tiled encoding is
        different from non-tiled encoding because each tile uses a different encoder. To avoid tiling artifacts, the
        tiles overlap and are blended together to form a smooth output. You may still see tile-sized changes in the
        output, but they should be much less noticeable.

        Args:
            x (`torch.FloatTensor`): Input batch of images/videos.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.autoencoder_kl.AutoencoderKLOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoder_kl.AutoencoderKLOutput`] or `tuple`:
                If return_dict is True, a [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned, otherwise a plain
                `tuple` is returned.
        """
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split video into tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[-2], overlap_size):
            row = []
            for j in range(0, x.shape[-1], overlap_size):
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_size,
                    j : j + self.tile_sample_min_size,
                ]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        moments = torch.cat(result_rows, dim=-2)
        if return_moments:
            return moments
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def spatial_tiled_decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Decode a batch of images/videos using a tiled decoder.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.
        """
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[-2], overlap_size):
            row = []
            for j in range(0, z.shape[-1], overlap_size):
                tile = z[
                    :,
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=-2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def temporal_tiled_encode(
        self, x: torch.FloatTensor
    ) -> DiagonalGaussianDistribution:
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_latent_min_tsize - blend_extent

        # Split the video into tiles and encode them separately.
        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i : i + self.tile_sample_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_sample_min_size
                or tile.shape[-2] > self.tile_sample_min_size
            ):
                tile = self.spatial_tiled_encode(tile, return_moments=True)
            else:
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])
        moments = torch.cat(result_row, dim=2)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def temporal_tiled_decode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        # Split z into overlapping tiles and decode them separately.

        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - blend_extent

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i : i + self.tile_latent_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (
                tile.shape[-1] > self.tile_latent_min_size
                or tile.shape[-2] > self.tile_latent_min_size
            ):
                decoded = self.spatial_tiled_decode(tile, return_dict=True).sample
            else:
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, : t_limit + 1, :, :])

        dec = torch.cat(result_row, dim=2)
        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    def encode_to_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into latents.
        Input comes in at (B,C,F,H,W) in the range (0,1)
        """
        z = self.encode(x, sample_posterior=True, return_posterior=False)
        return z

    def decode_from_latents(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Decodes latents into images."""
        return self.decode(z)

    def forward(
        self,
        batch: torch.FloatTensor,
        batch_idx=-1,
        optimizer_idx=-1,
        global_step=-1,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.FloatTensor, DiagonalGaussianDistribution, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = batch
        z, posterior = self.encode(
            x,
            return_posterior=True,
            sample_posterior=sample_posterior,
            generator=generator,
        )
        dec = self.decode(z)

        if optimizer_idx == -1:
            return dec, posterior

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(
                x,
                dec,
                posterior,
                optimizer_idx,
                global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            return aeloss, dec, posterior, log_dict_ae

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                x,
                dec,
                posterior,
                optimizer_idx,
                global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            return discloss, dec, posterior, log_dict_disc

    def get_last_layer(self):
        return self.decoder.conv_out.conv.weight

    def configure_optimizers(self, learning_rate):
        opt_ae = torch.optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quant_conv.parameters())
            + list(self.post_quant_conv.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.999),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc]

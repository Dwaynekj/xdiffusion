from collections import namedtuple
from einops import rearrange
import functools
import hashlib
import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import models
from torch_dwt.functional import dwt3


class LPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start,
        logvar_init=0.0,
        kl_weight=1.0,
        pixelloss_weight=1.0,
        disc_num_layers=3,
        disc_in_channels=3,
        adversarial_weight: float = 1.0,
        adversarial_start: int = 0,
        perceptual_weight=1.0,
        use_actnorm=False,
        disc_conditional=False,
        disc_loss="hinge",
        use_3d=False,
        use_reconstruction_gan=False,
        wavelet_loss_weight=0.0,
        rec_loss="l1",
        learned_logvar=True,
        use_nll=True,
        kl_start: int = 0,
        perceptual_start: int = 0,
        wavelet_start: int = 0,
        use_adaptive_adversarial_weight: bool = True,
    ):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.learned_logvar = learned_logvar

        # output log variance
        if learned_logvar:
            self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        if use_reconstruction_gan:
            self.discriminator = (
                NLayerDiscriminator(
                    input_nc=disc_in_channels * 2,
                    output_nc=2,
                    n_layers=disc_num_layers,
                    use_actnorm=use_actnorm,
                ).apply(weights_init)
                if not use_3d
                else NLayerDiscriminator3D(
                    input_nc=disc_in_channels * 2,
                    output_nc=2,
                    n_layers=disc_num_layers,
                    use_actnorm=use_actnorm,
                ).apply(weights_init)
            )
        else:
            self.discriminator = (
                NLayerDiscriminator(
                    input_nc=disc_in_channels,
                    n_layers=disc_num_layers,
                    use_actnorm=use_actnorm,
                ).apply(weights_init)
                if not use_3d
                else NLayerDiscriminator3D(
                    input_nc=disc_in_channels,
                    n_layers=disc_num_layers,
                    use_actnorm=use_actnorm,
                ).apply(weights_init)
            )
        self.wavelet_loss_weight = wavelet_loss_weight

        if wavelet_loss_weight > 0.0:
            assert use_3d
            self.wavelet_loss = WaveletLoss3D()

        self.rec_loss = rec_loss
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.adversarial_weight = adversarial_weight
        self.adversarial_start = adversarial_start
        self.disc_conditional = disc_conditional
        self.use_reconstruction_gan = use_reconstruction_gan
        self.use_nll = use_nll
        self.perceptual_start = perceptual_start
        self.kl_start = kl_start
        self.wavelet_start = wavelet_start
        self.use_adaptive_adversarial_weight = use_adaptive_adversarial_weight

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.adversarial_weight
        return d_weight

    def forward(
        self,
        inputs,
        reconstructions,
        posteriors,
        optimizer_idx,
        global_step,
        last_layer=None,
        cond=None,
        split="train",
        weights=None,
    ):
        if self.rec_loss == "l1":
            rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())
        else:
            assert self.rec_loss == "l2"
            rec_loss = (inputs.contiguous() - reconstructions.contiguous()) ** 2

        perceptual_weight = adopt_weight(self.perceptual_weight, global_step=global_step, threshold=self.perceptual_start)
        if perceptual_weight > 0:
            p_loss = self.perceptual_loss(
                inputs.contiguous(), reconstructions.contiguous()
            )
            rec_loss = rec_loss + perceptual_weight * p_loss

        wavelet_loss_weight = adopt_weight(self.wavelet_loss_weight, global_step=global_step, threshold=self.wavelet_start)
        if wavelet_loss_weight > 0.0:
            w_loss = self.wavelet_loss(
                reconstructions.contiguous(), inputs.contiguous()
            )
            rec_loss = rec_loss + wavelet_loss_weight * w_loss
        else:
            w_loss = torch.zeros_like(rec_loss)

        if self.learned_logvar:
            logvar = self.logvar
        else:
            logvar = torch.mean(posteriors.logvar, dim=-1, keepdim=True)
            logvar = torch.mean(logvar, dim=-2, keepdim=True)
            logvar = torch.mean(logvar, dim=-3, keepdim=True)
            logvar = torch.mean(logvar, dim=-4, keepdim=True)

        nll_loss = rec_loss / torch.exp(logvar) + logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional

                if self.use_reconstruction_gan:
                    # If using the reconstruction GAN loss, then we pass in both the
                    # reconstructed image and the real image. We don't use the real part
                    # of the discriminator in the G loss.
                    logits_real_fake = self.discriminator(
                        torch.cat(
                            [reconstructions.contiguous(), inputs.contiguous()], dim=1
                        ).contiguous()
                    )
                    logits_fake, _ = torch.chunk(logits_real_fake, 2, dim=1)
                else:
                    logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                assert not self.use_reconstruction_gan
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous(), cond), dim=1)
                )
            g_loss = -torch.mean(logits_fake)

            try:
                if self.use_adaptive_adversarial_weight:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                else:
                    d_weight = torch.ones_like(g_loss) * self.adversarial_weight
            except RuntimeError:
                assert not self.training
                d_weight = torch.zeros_like(g_loss)

            loss = (
                weighted_nll_loss if self.use_nll else torch.mean(rec_loss)
                + adopt_weight(self.kl_weight, global_step=global_step, threshold=self.kl_start) * kl_loss
                + adopt_weight(d_weight, global_step, threshold=self.adversarial_start) * g_loss
            )

            log = {
                "{}/total_loss".format(split): loss.clone().detach().mean().item(),
                "{}/logvar".format(split): logvar.detach().mean().item(),
                "{}/kl_loss".format(split): kl_loss.detach().mean().item(),
                "{}/nll_loss".format(split): nll_loss.detach().mean().item(),
                "{}/rec_loss".format(split): rec_loss.detach().mean().item(),
                "{}/d_weight".format(split): d_weight.detach().item(),
                "{}/g_loss".format(split): g_loss.detach().mean().item(),
                "{}/w_loss".format(split): w_loss.detach().mean().item(),
                "{}/w_loss_weight".format(split): self.wavelet_loss_weight,
            }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                if self.use_reconstruction_gan:
                    # If using the reconstruction GAN loss, then we pass in both the
                    # reconstructed image and the real image.
                    logits_fake_real = self.discriminator(
                        torch.cat(
                            [
                                reconstructions.contiguous().detach(),
                                inputs.contiguous().detach(),
                            ],
                            dim=1,
                        ).contiguous()
                    )
                    logits_fake_a, logits_real_a = torch.chunk(
                        logits_fake_real, 2, dim=1
                    )

                    logits_real_fake = self.discriminator(
                        torch.cat(
                            [
                                inputs.contiguous().detach(),
                                reconstructions.contiguous().detach(),
                            ],
                            dim=1,
                        ).contiguous()
                    )
                    logits_real_b, logits_fake_b = torch.chunk(
                        logits_real_fake, 2, dim=1
                    )
                    disc_loss = self.disc_loss(
                        logits_real_a, logits_fake_a
                    ) + self.disc_loss(logits_real_b, logits_fake_b)
                    logits_real = logits_real_a + logits_real_b
                    logits_fake = logits_fake_a + logits_fake_b
                else:
                    logits_real = self.discriminator(inputs.contiguous().detach())
                    logits_fake = self.discriminator(
                        reconstructions.contiguous().detach()
                    )
                    disc_loss = self.disc_loss(logits_real, logits_fake)
            else:
                assert not self.use_reconstruction_gan
                logits_real = self.discriminator(
                    torch.cat((inputs.contiguous().detach(), cond), dim=1)
                )
                logits_fake = self.discriminator(
                    torch.cat((reconstructions.contiguous().detach(), cond), dim=1)
                )
                disc_loss = self.disc_loss(logits_real, logits_fake)

            disc_factor = adopt_weight(
                1.0, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * disc_loss

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean().item(),
                "{}/logits_real".format(split): logits_real.detach().mean().item(),
                "{}/logits_fake".format(split): logits_fake.detach().mean().item(),
            }
            return d_loss, log


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class NLayerDiscriminator3D(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    Extended to 3D.
    """

    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator3D, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm3d
        else:
            assert False, "Activation norm not supported in 3d."

        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm3d
        else:
            use_bias = norm_layer != nn.BatchNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv3d(ndf * nf_mult, output_nc, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)


class ActNorm(nn.Module):
    def __init__(
        self, num_features, logdet=False, affine=True, allow_reverse_init=False
    ):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height * width * torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:, :, None, None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class WaveletLoss3D(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        l1_loss = torch.abs(
            dwt3(inputs.contiguous(), "haar") - dwt3(targets.contiguous(), "haar")
        )

        # Average over the number of wavelet filters, reducing the dimensions
        l1_loss = torch.mean(l1_loss, dim=1)

        # Average over all of the filter banks, keeping dimensions
        l1_loss = torch.mean(l1_loss, dim=-1, keepdim=True)
        l1_loss = torch.mean(l1_loss, dim=-2, keepdim=True)
        l1_loss = torch.mean(l1_loss, dim=-3, keepdim=True)
        return l1_loss


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1,
            requires_grad=False,
        )
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "lpips")
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name, "lpips")
        model.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        return model

    def forward(self, input, target):
        B = input.shape[0]
        num_dims = len(input.shape)

        # If the input is 3D, we need to move the temporal dimension into the
        # batch dimension.
        if num_dims == 5:
            assert len(target.shape) == 5
            input = rearrange(input, "b c f h w -> (b f) c h w")
            target = rearrange(target, "b c f h w -> (b f) c h w")

        # If the input is only single channel, repeat the channels until
        # they match the model input (3 channel)
        if input.shape[1] == 1:
            assert target.shape[1] == 1
            input = torch.tile(input, (1, 3, 1, 1))
            target = torch.tile(target, (1, 3, 1, 1))

        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(
                outs1[kk]
            )
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [
            spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
            for kk in range(len(self.chns))
        ]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]

        if num_dims == 5:
            # The batch channel contains the frame information.
            # Currently we are shape "(b f) 1 1 1". Convert this to
            # "b 1 f 1 1", then average the frame dimension.
            val = rearrange(val, "(b f) c h w -> b c f h w", b=B)
            val = torch.mean(val, dim=2, keepdim=True)

        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, weights=None):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(weights=weights).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real))
        + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


URL_MAP = {"vgg_lpips": "https://heibox.uni-heidelberg.de/f/607503859c864bc1b30b/?dl=1"}

CKPT_MAP = {"vgg_lpips": "vgg.pth"}

MD5_MAP = {"vgg_lpips": "d507d7349b931f0638a25a48a722f98a"}


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path

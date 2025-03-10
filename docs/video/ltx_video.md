# LTX-Video: Realtime Video Latent Diffusion

In this example we introduce LTX-Video, from the paper [LTX-Video: Realtime Video Latent Diffusion](https://arxiv.org/abs/2501.00103v1).

## Introduction

LTX-Video employs a holistic approach to the latent diffusion process, seamlessly integrating the Video-VAE and the denoising transformer, optimizing their interation within the compressed latent space and sharing the denoising objective between the transformer and the VAE's decoder.

The Video-VAE is a high compression VAE leveraging a novel loss function. It relocates the patchifying operation from the score network to the VAE, and achieves a 1:192 compression ratio with spatiotemporal downsampling of of 32x32x8.It achieves this by tweaking the standard VAE losses. It uses a novel "Reconstruction GAN" loss on the discriminator, which concatenates both the fake sample and the real sample and forces the discriminator to decide which is the original and which is the fake, rather than simply giving the discriminator either the fake or the real image. This simple conditioning creates a more powerful discriminator for reconstruction tasks like this (versus generative tasks), and forces the generator to get better in turn. In addition, the generator uses an additional discrete wavelet loss to help recover high frequency details, and uses per-layer injected noise in the decoder to improve overall diversity of high frequency details (although in our implementation we ended not using the per-layer decoder noise because we didn't have time to ablate it). In addition, the VAE acts as a denoising decoder, where the decoder is conditioned on the timestep of the diffusion process, and takes over the last timestep of the sampling process. This helps recover high frequency details according the paper. 

The LTX-Video score network is a transformer based latent diffusion model. It adopts the Pixart-α transformer architecture, which extends the original DiT transformer architecture with open text conditionings. LTX-Video adds the following extensions:

-   Replaces the traditional absolute position embeddings with RoPE embeddings enhanced by normalized fractional coordinates.
-   Query/Key normalization using RMSNorm
-   Uses cross-attention ala Pixart-α rather than MM-DiT from SD3

## Configuration File

The configuration file for the VAE is located in [LTX-Video VAE](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/video/moving_mnist/ltx_video/autoencoder.yaml).

The configuration file for the diffusion model is located in [LTX-Video](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/video/moving_mnist/ltx_video/ltx_video.yaml).

## Training

The first step is to train the VAE model in order to compress the pixel space videos into latent space. We successfully trained the VAE using a batch size of 8 for 100k steps on an A10.

```
> torchrun --nproc-per-node 1 --nnodes 1 training/video/autoencoder.py --config_path configs/video/moving_mnist/ltx_video/autoencoder.yaml
```

To train the LTX-Video model, use the following, with the pre-trained VAE checkpoint from the first step:

```
> torchrun --nproc-per-node 1 --nnodes 1 training/video/train.py --config_path configs/video/moving_mnist/ltx_video/ltx_video.yaml --load_vae_weights_from_checkpoint ltx_video_vae-100000.pt --num_training_steps 100000
```

We successfully tested training on a single A10 instance (24GB VRAM) using a batch size of 128 for 100k steps.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python sampling/video/moving_mnist/sample.py --config_path configs/video/moving_mnist/ltx_video/ltx_video.yaml --num_samples 8 --checkpoint output/video/moving_mnist/ltx_video/diffusion-100000.pt
```

Output will be saved to the `output/video/moving_mnist/sample/ltx_video` directory.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/moving_mnist/ltx_video/ltx_video.yaml) | [google drive](https://drive.google.com/file/d/1iNU2FsBv687FGAWkEU3bxZbrz_QJKafj/view?usp=sharing) | ![LTX-Video](https://drive.google.com/uc?export=view&id=1zOCNPirEWQkL_REV6pNFdakkBVcnDBrB)

## Other Resources

- Original LTX-Video github release: [Github](https://github.com/Lightricks/LTX-Video)

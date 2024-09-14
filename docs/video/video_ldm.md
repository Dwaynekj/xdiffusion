# Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models

In this example we introduce Video-LDM from the paper [Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2304.08818).

## Introduction

Video-LDM is a fun model that can turn any image diffusion model into a video diffusion model. The key insight they introduce is adding training temporal layers to a fixed, frozen image diffusion model, and treating the spatio-temporal data as spatial data in the batch dimension (so `"b c f h w -> (b f) c h w"`). This is cool because the model retains the ability to generate high quality imagery, and you are essentially just fine tuning for temporal relationships.

In this example, we train a standard v-parameterized DDPM Unet based model on the individual frames of Moving MNIST (see [Moving MNIST DDPM](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/moving_mnist/ddpm_32x32_v_continuous_clip.yaml) for the configuration file). This gives the model a strong prior to generate in-domain imagery. Then, we take those weights and train the video model on top of it.

One cool aspect of this model is that since the original image model weights are frozen, the model can generate high quality imagery at the beginning. So at the initial step, you will correctly generated frames, but with no temporal consistency. This differs from the using first step of an untrained model, which predicts random noise.

## Configuration File

The configuration file to train the image model is located in [Moving MNIST DDPM](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/moving_mnist/ddpm_32x32_v_continuous_clip.yaml)

The configuration file to train the video model is located in [Video LDM](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/video/moving_mnist/video_ldm.yaml).

## Training

First, ensure that all of the requirements are installed following the instructions [here](https://github.com/swookey-thinky/xdiffusion?tab=readme-ov-file#requirements).

Next, you need to train the base image diffusion model. To train the model, run the following from the root of the repository:

```
> python training/image/moving_mnist/train.py --config_path configs/image/moving_mnist/ddpm_32x32_v_continuous_clip.yaml --num_training_steps 20000
```

Now, we can train the video model using the above image model checkpoint. To train the model, run the following from the root of the repository:

```
> python training/video/moving_mnist/train.py --config_path configs/video/moving_mnist/video_ldm.yaml --batch_size 8 --load_model_weights_from_checkpoint output/image/moving_mnist/ddpm_32x32_v_continuous_clip/diffusion-20000.pt --num_training_steps 200000
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 8.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python training/video/moving_mnist/sample.py --config_path configs/video/moving_mnist/video_ldm.yaml --num_samples 16 --checkpoint output/video/moving_mnist/video_ldm/diffusion-100000.pt
```

Output will be saved to the `output/moving_mnist/sample/video_ldm` directory.

## Results and Checkpoints

The following results were generated after training on a single T4 instance for 100k steps at batch size 8:

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/video/moving_mnist/video_ldm.yaml) | [google drive](https://drive.google.com/file/d/17ItId0ogI00ELsMXTBUQAoaVKlsUyvya/view?usp=sharing) | ![Video LDM](https://drive.google.com/uc?export=view&id=1UsKoMKyaeQspVGxNhhowWx7OQ57XEIiH)


## Other Resources

Unfortunately the authors did not release their source code for this model.

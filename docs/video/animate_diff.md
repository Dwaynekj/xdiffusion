# AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning

In this example we introduce Animate-Diff from the paper [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning](https://arxiv.org/abs/2307.04725).

## Introduction

Animate-Diff is a very similar model and methodology to [Video-LDM](https://arxiv.org/abs/2304.08818), in that both of them can turn any pretrained image diffusion model into a video difusion model. Animate-Diff goes a step further and introduces the notion of Motion-LORA's, which we will explore in a separate lesson. The main architectural difference is that Video-LDM inserts 3D temporal convolutions in addition to temporal cross-attention, whereas AnimateDiff adds a temporal transformer module, with no cross attention to the text embeddings.

In this example, we train a standard v-parameterized DDPM Unet based model on the individual frames of Moving MNIST (see [Moving MNIST DDPM](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/moving_mnist/ddpm_32x32_v_continuous_clip.yaml) for the configuration file). This gives the model a strong prior to generate in-domain imagery. Then, we take those weights and train the video model on top of it.

One cool aspect of this model is that since the original image model weights are frozen, the model can generate high quality imagery at the beginning. So at the initial step, you will see correctly generated frames, but with no temporal consistency. This differs from the usual first step of an untrained model, which predicts random noise.

## Configuration File

The configuration file to train the image model is located in [Moving MNIST DDPM](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/moving_mnist/ddpm_32x32_v_continuous_clip.yaml)

The configuration file to train the video model is located in [AnimateDiff](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/video/moving_mnist/animate_diff.yaml).

## Training

First, ensure that all of the requirements are installed following the instructions [here](https://github.com/swookey-thinky/xdiffusion?tab=readme-ov-file#requirements).

Next, you need to train the base image diffusion model. To train the model, run the following from the root of the repository:

```
> python training/image/moving_mnist/train.py --config_path configs/image/moving_mnist/ddpm_32x32_v_continuous_clip.yaml --num_training_steps 20000
```

Now, we can train the video model using the above image model checkpoint. To train the model, run the following from the root of the repository:

```
> python training/video/moving_mnist/train.py --config_path configs/video/moving_mnist/animate_diff.yaml --batch_size 8 --load_model_weights_from_checkpoint output/image/moving_mnist/ddpm_32x32_v_continuous_clip/diffusion-20000.pt --num_training_steps 100000
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 8.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python training/video/moving_mnist/sample.py --config_path configs/video/moving_mnist/animate_diff.yaml --num_samples 16 --checkpoint output/video/moving_mnist/animate_diff/diffusion-100000.pt
```

Output will be saved to the `output/moving_mnist/sample/animate_diff` directory.

## Results and Checkpoints

The following results were generated after training on a single T4 instance for 100k steps at batch size 8:

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/video/moving_mnist/animate_diff.yaml) | [google drive](https://drive.google.com/file/d/1FlyvW7g1GhU5XqHKqaHMoafnr0hdCuXr/view?usp=sharing) | ![Video LDM](https://drive.google.com/uc?export=view&id=11EJWvEilKqmrGabCRvjM5CxW6I7mWEEg)

## Other Resources

The authors released the original source code for their model at [AnimateDiff](https://github.com/guoyww/AnimateDiff/)
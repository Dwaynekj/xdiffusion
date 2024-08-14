# SoRA: Video generation models as world simulators

In this example we introduce Sora, announecd in the blog post [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/).

## Introduction

Since there is no official SoRA code or paper, this model is based on what little information we can glean from public sources. We use two sources of information for the model we built here:

- [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/)
- [OpenSoRA](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_01.md)

From the first source, we know that the SoRA model has the following characteristics:

- Is a text-to-video model ("...we train text-conditional diffusion models...")
- Trained jointly on images and videos ("...jointly on videos and images of variable durations, resolutions and aspect ratios.")
- Uses a latent space representation ("...we turn videos into patches by first compressing videos into a lower-dimensional latent space")
- Decomposes the latent space into spacetime patches ("subsequently decomposing the representation into spacetime patches")
- Uses a spacetime transformer architecture ("Importantly, Sora is a diffusion transformer.")

Given the above information, we are going to start with the spacetime diffusion formulation introduced in [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048), and specifically we will use variation 3 of Latte, which incorporates the factorized spacetime attention blocks into a single transformer block, based on the text conditional diffusion transformer blocks from  [PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://arxiv.org/abs/2310.00426).

## Configuration File

The configuration file is located in [Flexible Diffusion Models](https://github.com/swookey-thinky/video_diffusion/blob/main/configs/moving_mnist/flexible_diffusion_models.yaml).

## Training

To train the video diffusion model, use:

```
> python training/moving_mnist/train.py --config_path configs/moving_mnist/flexible_diffusion_models.yaml
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 8.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python training/moving_mnist/sample.py --config_path configs/moving_mnist/flexible_diffusion_models.yaml --num_samples 8 --checkpoint output/moving_mnist/flexible_diffusion_models/diffusion-100000.pt
```

Output will be saved to the `output/moving_mnist/sample/flexible_diffusion_models` directory.

However, you can also autoregressively sample arbitrary length videos, using a defined sampling scheme. For example, in `configs/sampling_schemes/autoregressive.yaml`, we have defined an autoregressive sampling scheme that generates a 160 frame video, conditioning each segment on the previous 8 frames in the video. To run this, run:

```
> python training/moving_mnist/sample.py --config_path configs/moving_mnist/flexible_diffusion_models.yaml --num_samples 16 --checkpoint output/moving_mnist/flexible_diffusion_models/diffusion-100000.pt --sampling_scheme_path configs/sampling_schemes/autoregressive.yaml
```

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/video_diffusion/blob/main/configs/moving_mnist/flexible_diffusion_models.yaml) | [google drive](https://drive.google.com/file/d/1rDX-sioy4B3uUFjQfQnmZ5ASzIE7V5gb/view?usp=sharing) | ![Flexible Diffusion Models](https://drive.google.com/uc?export=view&id=1B2raR3_suRf8qAUP4jzi8YIwka-UHwrU)

## Other Resources

The authors released their original source code [here](https://github.com/plai-group/flexible-video-diffusion-modeling).

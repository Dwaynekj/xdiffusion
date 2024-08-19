# SoRA: Video generation models as world simulators

In this example we introduce Sora, announecd in the blog post [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/).

## Introduction

Since there is no official SoRA code or paper, this model is based on what little information we can glean from public sources. We use three sources of information for the model we built here:

- [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/)
- [OpenSoRA](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_01.md)
- [Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models] (https://arxiv.org/abs/2402.17177)

From the first source, we know that the SoRA model has the following characteristics:

- Is a text-to-video model ("...we train text-conditional diffusion models...")
- Trained jointly on images and videos ("...jointly on videos and images of variable durations, resolutions and aspect ratios.")
- Uses a latent space representation ("...we turn videos into patches by first compressing videos into a lower-dimensional latent space")
- Decomposes the latent space into spacetime patches ("subsequently decomposing the representation into spacetime patches")
- Uses a spacetime transformer architecture ("Importantly, Sora is a diffusion transformer.")

Given the above information, we are going to start with the spacetime diffusion formulation introduced in [Latte: Latent Diffusion Transformer for Video Generation](https://arxiv.org/abs/2401.03048), and specifically we will use variation 3 of Latte, which incorporates the factorized spacetime attention blocks into a single transformer block, based on the text conditional diffusion transformer blocks from  [PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://arxiv.org/abs/2310.00426).

## Configuration File

The configuration file is located in [SORA](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/video/moving_mnist/sora.yaml).

## Training

To train the SORA model, use:

```
> python training/video/moving_mnist/train.py --config_path configs/video/moving_mnist/sora.yaml
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 8.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python sampling/video/moving_mnist/sample.py --config_path configs/video/moving_mnist/sora.yaml --num_samples 8 --checkpoint output/videos/moving_mnist/sora/diffusion-100000.pt
```

Output will be saved to the `output/video/moving_mnist/sample/sora` directory.

However, you can also autoregressively sample arbitrary length videos, using a defined sampling scheme. For example, in `configs/video/sampling_schemes/autoregressive.yaml`, we have defined an autoregressive sampling scheme that generates a 160 frame video, conditioning each segment on the previous 4 frames in the video. To run this, run:

```
> python sampling/video/moving_mnist/sample.py --config_path configs/video/moving_mnist/sora.yaml --num_samples 16 --checkpoint output/video/moving_mnist/sora/diffusion-100000.pt --sampling_scheme_path configs/video/sampling_schemes/autoregressive.yaml
```

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/moving_mnist/sora.yaml) | [google drive](https://drive.google.com/file/d/1iNU2FsBv687FGAWkEU3bxZbrz_QJKafj/view?usp=sharing) | ![SoRA](https://drive.google.com/uc?export=view&id=1W40uY3xU5F36YTou-RT26Q6YidmrcJsp)

## Other Resources

- [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/)
- [OpenSoRA](https://github.com/hpcaitech/Open-Sora/blob/main/docs/report_01.md)
- [Sora: A Review on Background, Technology, Limitations, and Opportunities of Large Vision Models] (https://arxiv.org/abs/2402.17177)

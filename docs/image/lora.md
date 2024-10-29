# LoRA: Low-Rank Adaptation of Large Language Models

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to learn how to apply the technique of LoRA's from the language model paper [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) to image diffusion models.

## Introduction 

Low-Rank adaptation is a very cool technique to fine-tune an image diffusion model without training all of the weights of the model. Instead, we train only a very small number of weights, which are used to adapt the residuals of the model.

There is a very good introduction to this technique in the [cloneofsimo](https://github.com/cloneofsimo/lora) repository, which introduced the LoRA's for diffusion model. We will summarize it here:

Well, what's the alternative? In the domain of LLM, researchers have developed Efficient fine-tuning methods. LoRA, especially, tackles the very problem the community currently has: end users with Open-sourced stable-diffusion model want to try various other fine-tuned model that is created by the community, but the model is too large to download and use. LoRA instead attempts to fine-tune the "residual" of the model instead of the entire model: i.e., train the `ΔW` instead of `W`.

<p align=center>W' = W + ΔW</p>

Where we can further decompose `ΔW` into low-rank matrices: `ΔW = AB^T`, where `A ∈ R^(n×d)`,`B ∈ R^(m×d)`, `d<<n`. This is the key idea of LoRA. We can then fine-tune `A` and `B` instead of `W`. In the end, you get an insanely small model as `A` and `B` are much smaller than `W`.

Another nice feature is that the authors of the paper found you typically only need to adapt the attention modules in the model, which yields even smaller model sizes.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Configuration File

There is no specific on configuration file for training a LoRA. Instead, we use a separate training script, detailed below.

## Training

To train a LoRA model, use with a pretrained checkpoint from any other model:

```
> python training/image/train.py --config_path configs/image/mnist/ddpm_32x32_v_continuous_clip.yaml --dataset_name "image/mnist_inverted" --mixed_precision bf16 --use_lora_training --load_model_weights_from_checkpoint output/image/mnist/ddpm_32x32_v_continuous_clip/diffusion-10000.pt
```

Note that the training script or LoRA's will be trying to learn an inverted style for MNIST. So where the original model will output white text on a black background, applying the LoRA weights to that same, frozen model will output blak text on a white background.

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 128.

## Sampling

To sample from a pretrained checkpoint with LoRA weights, you can run:

```
> python sampling/image/mnist/sample.py --config_path configs/image/mnist/ddpm_32x32_v_continuous_clip.yaml --num_samples 8 --checkpoint output/image/mnist/ddpm_32x32_v_continuous_clip/diffusion-10000.pt --lora_path output/image/moving_mnist/lora/ddpm_32x32_v_continuous_clip/difusion-20000-lora.pt
```

Output will be saved to the `output/image/mnist/sample/ddpm_32x32_v_continuous_clip` directory.

## Results and Checkpoints

| Config | Checkpoint | Original Model | Applied LoRA Weights
| ------ | ---------- | ------- | -------
| [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/ddpm_32x32_v_continuous_clip.yaml) | [google drive](https://drive.google.com/file/d/1atzhtv-kRegnabROGZs6olxuVONiRQKI/view?usp=sharing) | ![LoRA Original](https://drive.google.com/uc?export=view&id=1_r8poe1SJxf8UtT4mmQaTT378m26hD-F) | ![LoRA](https://drive.google.com/uc?export=view&id=1NGtmYiLNAtOTC46UK7nbpkGf3NYfSVWI)


## Other Resources

The first repo to apply low-rank adaptation to diffusion model is located [here](https://github.com/cloneofsimo/lora).

LoRA support has also been added to huggingface and the [diffusers](https://github.com/huggingface/diffusers) library, which you can read about [here](https://huggingface.co/docs/diffusers/main/en/training/lora)

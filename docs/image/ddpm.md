# Denoising Diffusion Probabilistic Models (DDPM)

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to learn about the seminal image diffusion model from  [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239).

## Introduction

This paper really laid the foundation for all of the generative models to come, and was the first to bring together the theory from [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) and [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) and  give them a more modern and flexible framework. Importantly, this paper introduced the usage of sinusoidal position embeddings for conditioning on the timestep (based on modern transformers), and adding attention into the noise prediction network. 

If you want to look at the author's original codebase, you can find it [here](https://github.com/hojonathanho/diffusion). The original code is written in Tensorflow, but it easy to follow along with.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements) to set up your environment.

## Configuration File

The configuration file is located in [Imagen](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/ddpm_32x32_epsilon_discrete.yaml).

## Training

To train the Imagen model, use:

```
> python training/videoimage/mnist/train.py --config_path configs/image/mnist/ddpm_32x32_epsilon_discrete.yaml
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 128.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python sampling/image/mnist/sample.py --config_path configs/image/mnist/ddpm_32x32_epsilon_discrete.yaml --num_samples 8 --checkpoint output/image/mnist/ddpm_32x32_epsilon_discrete.yaml/diffusion-10000.pt
```

Output will be saved to the `output/image/mnist/sample/ddpm_32x32_epsilon_discrete.yaml` directory.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/ddpm.yaml) | https://drive.google.com/file/d/1OvpVLQCEsznQ7JMpqey-sxsm1Dsn96PL/view?usp=sharing | ![DDPM](https://drive.google.com/uc?export=view&id=1Yd8hhK9EhFMhfqQJf3CjAtFqK_XdZPSi)

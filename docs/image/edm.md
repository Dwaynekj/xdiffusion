# Elucidating the Design Space of Diffusion-Based Generative Models


[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to learn about an improved and unified diffusion formulation from [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364).

## Introduction

This paper is a fascinating look at ablating a lot of the design decisions from previous work, including DDPM++ from [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), NCSN++ from [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456), ADM from [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233), and iDDPM from [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672). Using the architectures from the above papers, the authors propose a new loss formulation, model preconditioning, augmentations, and sampling algorithm to improve upon the results of the previous authors.

The original source code for the paper was published at [EDM](https://github.com/NVlabs/edm/).

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/xdiffusion?tab=readme-ov-file#requirements) to set up your environment.

## Configuration File

There are several different configurations you can experiment with:

| Config | Description |
| ------ | ----------- |
| [EDM](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/edm.yaml) | Full EDM preconditioning and loss with DDPM++ architecture. |

## Training

To train the basic EDM model, use:

```
> python training/image/train.py --config_path configs/image/mnist/edm.yaml --dataset_name "image/mnist"
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 64.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python sampling/image/mnist/sample.py --config_path configs/image/mnist/edm.yaml --num_samples 8 --checkpoint output/image/mnist/edm/diffusion-20000.pt
```

Output will be saved to the `output/image/mnist/sample/edm` directory.

## Results and Checkpoints

| Config | Checkpoint | Num Sampling Steps | Results
| ------ | ---------- | ------- | -------
| [EDM](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/edm.yaml) | [Google Drive](https://drive.google.com/file/d/1lAatYJKvetBaOhYiioeX2YoA6OTxymxl/view?usp=sharing) | 18 | ![EDM](https://drive.google.com/uc?export=view&id=1yUeR5ep9mK1IwMsTyHwhyAqlftFwBNYz)

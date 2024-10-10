# Diffusion Models Without Attention

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to learn about the DiffuSSM diffusion model from [Diffusion Models Without Attention](https://arxiv.org/abs/2311.18257).

## Introduction

One of the key computational constraints with many diffusion model implementations is the quadratic complexity of the attention operation. Most recent diffusion models use a transformer backbone, and so the quadratic complexity in token length quickly starts to limit the size of the models that can be used, and the number of tokens that can be processed at any one time. In language modeling, and interesting area of research is State Space Models, introduced in [Efficiently modeling long sequences with structured state spaces](https://arxiv.org/abs/2111.00396), which aims to learn gneralizable sequence models with long range dependencies without incurring the quadratic cost of attention operations.

In this paper, the authors use the SSM formulation from language modeling and apply it to the image diffusion modeling task. 

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/xdiffusion?tab=readme-ov-file#requirements) to set up your environment.

## Configuration File

The configuration file is located in [DiffuSSM](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/diffussm.yaml).

## Training

To train the DiffuSSM model, use:

```
> python training/image/mnist/train.py --config_path configs/image/mnist/diffussm.yaml
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 8.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python sampling/image/mnist/sample.py --config_path configs/image/mnist/diffussm.yaml --num_samples 8 --checkpoint output/image/mnist/diffussm/diffusion-10000.pt
```

Output will be saved to the `output/image/mnist/sample/diffussm` directory.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/diffussm.yaml) | [Google Drive](https://drive.google.com/file/d/1i4jFXRS4enCexO4Zvgw_l54AfhYyo7Y9/view?usp=sharing) | ![DiffuSSM](https://drive.google.com/uc?export=view&id=1YKO1JUGeW9DbzqtGtA3HAz_gcNLKnBlZ)

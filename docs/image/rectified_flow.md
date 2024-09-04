# Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to learn about a rectified flow diffusion model from [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003).

## Introduction

Rectified flows are an alternative ODE formulation that is simple and yet yields surprisingly good results. For diffusion models, the code is formulated based on the Score SDE formulation from [Score-Based Generative Modeling through Stochastic Differential Equations] (https://arxiv.org/abs/2011.13456). For our purposes, this is an exciting paper to study because it is used as part of the innovation in the Stable Diffusion 3.0 model from the paper [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).

The original source code for the paper was published at [RectifiedFlow](https://github.com/gnobitab/RectifiedFlow).

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/xdiffusion?tab=readme-ov-file#requirements) to set up your environment.

## Configuration File

The configuration file is located in [Rectified Flow](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/rectified_flow_32x32.yaml).

## Training

To train the rectified flow model, use:

```
> python training/image/mnist/train.py --config_path configs/image/mnist/rectified_flow_32x32.yaml
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 128.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python sampling/image/mnist/sample.py --config_path configs/image/mnist/rectified_flow_32x32.yaml --num_samples 8 --checkpoint output/image/mnist/rectified_flow_32x32/diffusion-20000.pt
```

Output will be saved to the `output/image/mnist/sample/rectified_flow_32x32` directory.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/rectified_flow_32x32.yaml) | https://drive.google.com/file/d/191OKe-j65V4hkN4MciAv-2Uiszr3Fq7s/view?usp=sharing | ![Rectified Flow](https://drive.google.com/uc?export=view&id=14TOqFXSWiFpeUVnDuMfLRcRDUV5onuKQ)

After training the network for 10k steps, the full unconditional rectified flow model is able to generate samples like the below:

![Rectified Flow](https://drive.google.com/uc?export=view&id=14TOqFXSWiFpeUVnDuMfLRcRDUV5onuKQ)


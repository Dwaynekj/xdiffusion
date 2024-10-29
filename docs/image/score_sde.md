# Score-Based Generative Modeling through Stochastic Differential Equations

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to learn about formulating diffusion models with stochastic differential equations from [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456).

## Introduction

In this lesson, we are going to look at a generalization to the theory of all prior lessons, [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456). The authors will even derive the results for both [NSCN](https://arxiv.org/abs/1907.05600) and [DDPM](https://arxiv.org/abs/2006.11239) from this new unified theory, which is pretty cool!

The author's approach this problem as the solution to a generalized stochastic differential equation (SDE), and in particular, the reverse time solution to that same SDE. They propose several different SDE's for the diffusion process, as well as new samplers based on the long history of SDE solvers in other domains.

You can find the original FLAX source code from the authors is [here](https://github.com/yang-song/score_sde/tree/main). They also have a PyTorch implementation [here](https://github.com/yang-song/score_sde_pytorch).

We will implement several different variations of the SDE's introduced in the paper, as well as a few different variations of the new sampling techniques.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/xdiffusion?tab=readme-ov-file#requirements) to set up your environment.

## Configuration File

There are several different configurations you can experiment with:

| Config | Description |
| ------ | ----------- |
| [Variance Preserving SDE](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/score_sde_vpsde_discrete.yaml) | Discrete time formulation of the variance preserving SDE (theoretically equivalent to the original DDPM paper). |
| [Variance Preserving SDE - Continuous](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/score_sde_vpsde_continuous.yaml) | Continuous time formulation of the variance preserving SDE, with an Euler-Maruyama sampler. |
| [Sub-Variance Preserving SDE](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/score_sde_subvpsde.yaml) | Sub-variance preserving SDE, with a Predictor-Corrector sampler. |

## Training

To train the basic VPSDE, use:

```
> python training/image/train.py --config_path configs/image/mnist/score_sde_vpsde_discrete.yaml --dataset_name "image/mnist"
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 128.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python sampling/image/mnist/sample.py --config_path configs/image/mnist/score_sde_vpsde_disrete.yaml --num_samples 8 --checkpoint output/image/mnist/score_sde_vpsde_discrete/diffusion-10000.pt
```

Output will be saved to the `output/image/mnist/sample/score_sde_vpsde_discrete` directory.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [VPSDE - Discrete](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/score_sde_vpsde_discrete.yaml) | [Google Drive](https://drive.google.com/file/d/1gnrUM1Ecg37eN18oCDs3dvsm7XTbRmJ3/view?usp=sharing) | ![VPSDE Discrete](https://drive.google.com/uc?export=view&id=1GXjDhpFdSEg8wo0maCxGMWIsfCtkOPXY)
| [VPSDE - Continuous](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/score_sde_vpsde_continuous.yaml) | [Google Drive](https://drive.google.com/file/d/1YeEwu6YOEZlNI14hq4_kR3WWZLJeBgr2/view?usp=sharing) | ![VPSDE Continuous](https://drive.google.com/uc?export=view&id=1WVRJtlOMZwJ4KmvbDSFS_TjHr_he4MCO)
| [Sub-VPSDE](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/score_sde_subvpsde.yaml) | [Google Drive](https://drive.google.com/file/d/1jlcECpJsjwpM12yYJpzIp2jsRO7qgEeO/view?usp=sharing) | ![Sub-VPE SDE](https://drive.google.com/uc?export=view&id=1SPCFu0aFkcfXKpmLE2p3RteIZ-dCBeuA)

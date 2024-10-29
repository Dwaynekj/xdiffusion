# Consistency Models


[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to learn new class of generative models (in addition to a new distillation technique) called Consistency Models from [Consistency Models](https://arxiv.org/abs/2303.01469).

## Introduction

Consistency models are both a distillation technique as well as a new class of generative models. The concept is fairly simple. Using the SDE/ODE formulation of diffusion models, whereby any SDE can be interpreted as a probability flow ODE, the consistency model formulation enforces the *consistency* property during training/distillation, which states that all points along the same probability flow ODE trajectory map to the same initial point. Given this stipulation, this naturally leads to a **one-step** sampling method with these models, which is incredibly cool!

In this repository, we demonstrate both the training (where we train a consistency model from scratch) and distillation aspects of consistency models.

The original source code for the paper was published at [Consistency Models](https://github.com/openai/consistency_models/).

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/xdiffusion?tab=readme-ov-file#requirements) to set up your environment.

## Configuration File

| Config | Description |
| ------ | ----------- |
| [Consistency Model](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/consistency_model.yaml) | Training from scratch or distilling an existing model. |


## Training

To train the basic consistency model from scratch, use:

```
> python training/image/train.py --config_path configs/image/mnist/consistency_model.yaml --dataset_name "image/mnist"
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 64.

## Sampling

To sample from a pretrained checkpoint, using the 1-step sampler, you can run:

```
> python sampling/image/mnist/sample.py --config_path configs/image/mnist/consistency_model.yaml --num_samples 64 --checkpoint output/image/mnist/consistency_model/diffusion-100000.pt --sampler_config_path configs/image/mnist/sampler/consistency_model_onestep.yaml
```

Output will be saved to the `output/image/mnist/samplers/consistency_model` directory.

## Results and Checkpoints

| Config | Checkpoint | Num Sampling Steps | Results
| ------ | ---------- | ------- | -------
| [Consistency Model](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/consistency_model.yaml) | [Google Drive](https://drive.google.com/file/d/1iT2RxA7yJs2udO2qQDv8fkSe5JcwARTn/view?usp=sharing) | 1 | ![1](https://drive.google.com/uc?export=view&id=12hMpGtyLrfTy0BdJMSQ4GmPdyFd4ceE4)
| [Consistency Model](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/consistency_model.yaml) | [Google Drive](https://drive.google.com/file/d/1iT2RxA7yJs2udO2qQDv8fkSe5JcwARTn/view?usp=sharing) | 3 | ![3](https://drive.google.com/uc?export=view&id=12wVUP7Gid2-mzHj0gAgPpuVgsOvIQahU)
| [Consistency Model](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/consistency_model.yaml) | [Google Drive](https://drive.google.com/file/d/1iT2RxA7yJs2udO2qQDv8fkSe5JcwARTn/view?usp=sharing) | 40 | ![40](https://drive.google.com/uc?export=view&id=1Zgj38dDdEwGvHKFMJ0zgR37zrA5fQ-vx)

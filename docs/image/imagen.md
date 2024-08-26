# Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen)

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to learn about the Imagen diffusion model from [Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487).

## Introduction

Imagen posits that a larger language model (deep language understanding) yields better results than scaling other factors (model size, embedding size, etc). In Imagen, the authors used a T5 XXL language model, in conjunction with a cascade of diffusion models - a base  64x64 resolution diffusion model followed by a 64->256 diffusion super-resolution model and a final 256->1024 diffusion super-resolution model. Some of the attributes of Imagen include:

1. T5 XXL text encoder
2. Cascade of diffusion models
3. Classifier free guidance with large guidance weights
4. Static/dynamic thresholding of x_hat predictions in sampling
5. Gaussian conditioning augmentation and conditioning on the augmentation level in the super-resolution models
6. Base network uses the Improved DDPM architecture
7. Text embeddings are added to the timestep conditioning via a pooled embedding vector, as well as at multiple resolution using cross attention from Latent Diffusion Models. LayerNorm at the attention and pooling layers helped as well.
8. Improved UNet architecture for the super-resolution models (Efficient UNet)

We've also introduced a new code structure in this lesson. Since all of the lessons are building off each, and are essentially "picking and choosing" different pieces, we merged 
the implementations of all of the lessons and made it easier to configure them through YAML files. So now you will see the main details of the lesson in the YAML files, and the individual additional pieces added in code.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/mindiffusion?tab=readme-ov-file#requirements) to set up your environment.

## Configuration File

The configuration file is located in [Imagen](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/imagen.yaml).

## Training

To train the Imagen model, use:

```
> python training/videoimage/mnist/train.py --config_path configs/image/mnist/imagen.yaml
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 64.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python sampling/image/mnist/sample.py --config_path configs/image/mnist/imagen.yaml --num_samples 8 --checkpoint output/image/mnist/imagen/diffusion-10000.pt
```

Output will be saved to the `output/image/mnist/sample/imagen` directory.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/imagen.yaml) | https://drive.google.com/file/d/1HsadwpD94Vy6qyf_jlBaZn2jwlkREwPw/view?usp=sharing | ![Imagen](https://drive.google.com/uc?export=view&id=1MKyRgPKoPRFHLzd78aTA1K3QHgNm08Px)


After training the network for 10k steps, the full Imagen pipeline is able to generate samples like the below:

![Imagen](https://drive.google.com/uc?export=view&id=1MKyRgPKoPRFHLzd78aTA1K3QHgNm08Px)

 The prompts we used for generation above were:

<pre>
7 one five seven 1 4 nine two 
8 2 nine 8 two one nine three 
0 nine eight 5 6 zero 2 five 
eight three 2 8 2 two 7 one 
7 seven 7 0 nine zero seven 3 
nine nine one 2 six 8 eight three 
eight 2 5 nine 8 five five nine 
nine 0 five 3 7 four 8 eight 
</pre>

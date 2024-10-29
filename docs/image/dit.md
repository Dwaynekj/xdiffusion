# Scalable Diffusion Models with Transformers (DiT)

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to learn about the DiT diffusion model from [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748).

## Introduction

This model is interesting because it was the first to replace the UNet architecture in the diffusion score network with a transformer based architecture, from [Attention is all you need](https://arxiv.org/abs/1706.03762), and in particular uses the Vision Transformer architecture from [An image is worth 16x16 words: Transformers for image recognition at scale](https://arxiv.org/abs/2010.11929) as the backbone for the score network.

The original DiT paper used transformers to train a latent diffusion model, but for simplicity, we will be training in pixel space rather than latent space. Note that this is a class conditional model as well, without text alignment (for now). Future research uses transformer backbones with text alignment so we will explore that there.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/xdiffusion?tab=readme-ov-file#requirements) to set up your environment.

## Configuration File

The configuration file is located in [DiT](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/dit.yaml).

## Training

To train the DiT model, use:

```
> python training/image/train.py --config_path configs/image/mnist/dit.yaml --dataset_name "image/mnist"
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 128.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python sampling/image/mnist/sample.py --config_path configs/image/mnist/dit.yaml --num_samples 8 --checkpoint output/image/mnist/dit/diffusion-10000.pt
```

Output will be saved to the `output/image/mnist/sample/dit` directory.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/dit.yaml) | https://drive.google.com/file/d/1cPhUH78AFALLdmhv1gtddQQtZd_1MbbA/view?usp=sharing | ![DiT](https://drive.google.com/uc?export=view&id=1J6ktzFr7iqgWcf23JpgVaM81Z7sUUcmj)


After training the prior network for 30k steps, the full Dit pipeline is able to generate samples like the below:

 ![DiT](https://drive.google.com/uc?export=view&id=1J6ktzFr7iqgWcf23JpgVaM81Z7sUUcmj)

 The prompts we used for generation above were:

<pre>
2 4 one nine 2 eight 7 9 
two two 2 eight nine 6 nine one 
nine 4 seven 2 one two four 1 
eight two two two six six eight nine 
five 1 seven 0 4 seven four 2 
one 0 five 9 five 4 5 four 
1 two five 4 9 0 zero one 
1 4 eight seven 8 eight zero 8 
</pre>

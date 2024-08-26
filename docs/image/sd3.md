# Stable Diffusion 3 - Scaling Rectified Flow Transformers for High-Resolution Image Synthesis

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to learn about the Stable Diffusion 3 diffusion model from Stability AI. This model was introduced in the technical report [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206).

## Introduction
Stable Diffusion 3 is a very cool model that combines a lot of recent research about image diffusion models. It extends the diffusion transformer network from [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) by separating out the text and image handling using a multimodel, dual stream approach. It also uses a recently introduced rectified flow formulation which builds on flow matching from [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).

In our implementation here, we train in pixel space rather than the latent space of the original model, and we scale down the transformer network to ~50m parameters.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/xdiffusion?tab=readme-ov-file#requirements) to set up your environment.

## Configuration File

The configuration file is located in [SD3](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/sd3.yaml).

## Training

To train the SD3 model, use:

```
> python training/videoimage/mnist/train.py --config_path configs/image/mnist/sd3.yaml
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 64.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python sampling/image/mnist/sample.py --config_path configs/image/mnist/sd3.yaml --num_samples 8 --checkpoint output/image/mnist/sd3/diffusion-20000.pt
```

Output will be saved to the `output/image/mnist/sample/sd3` directory.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/sd3.yaml) | https://drive.google.com/file/d/1xzSianm9pdgkhEJPVwtvCrJm3aN7EE2J/view?usp=sharing | ![SD3](https://drive.google.com/uc?export=view&id=1YI6iezQHbyAKiyyChnyD6_8KQaPdxIxn)


After training the network for 20k steps at batch size 64, the sd3 model pipeline is able to generate samples like the below:

![SD3](https://drive.google.com/uc?export=view&id=1YI6iezQHbyAKiyyChnyD6_8KQaPdxIxn)

The prompts we used for generation above were:

<pre>
1 zero 1 9 five seven 1 0 
zero seven 7 2 5 7 1 five 
3 6 8 four two seven 1 2 
nine eight 9 1 nine four 7 three 
four three 4 two 5 zero eight 7 
four 6 2 4 9 0 0 one 
six 2 2 5 1 zero two seven 
eight 2 8 two 2 three zero five 
</pre>

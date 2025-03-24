# Dynamic Tanh

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to rebuild several different diffusion models using the results from Meta's [Transformers Without Normalization](https://arxiv.org/abs/2503.10622) paper.

## Introduction 

[Transformers Without Normalization](https://arxiv.org/abs/2503.10622) introduces a simple replacement for LayerNorm inside of general transformer architectures. The paper isn't specifically about diffusion models, but we thought it would be fun to try this out on transformer based diffusion models and see how it does. In particular, we look at both PixArt-α from [PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://arxiv.org/abs/2310.00426)
and FLUX from [Black Forest Labs]([here](https://github.com/black-forest-labs/flux)), where we replace all of the LayerNorm modules with Dynamic Tanh modules.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Configuration File

The configuration file is located in [PixArt-α DyT](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/pixart_alpha_dyt.yaml) and [Flux DyT](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/flux_dyt.yaml).

## Training

To train the PixArt-α model, use:

```
> python training/image/train.py --config_path configs/image/mnist/pixart_alpha_dyt.yaml --num_training_steps 20000
```

We successfully tested training on a single GH200 instance from Lambda (96GB VRAM) using a batch size of 1024 and 20k training steps, for a total cost of $2.88.

To train the flux version of the model, use:

```
> python training/image/train.py --config_path configs/image/mnist/flux_dyt.yaml --num_training_steps 10000
```

We successfully tested training on a single GH200 instance from Lambda (96GB VRAM) using a batch size of 1024 and 10k training steps, for a total cost of $4.20.


## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python sampling/image/mnist/sample.py --config_path configs/image/mnist/flux_dyt.yaml --num_samples 8 --checkpoint output/image/mnist/flux_dyt/diffusion-10000.pt
```

Output will be saved to the `output/image/mnist/sample/flux_dyt` directory.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [PixArt-α](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/pixart_alpha_dyt.yaml) | [google drive](https://drive.google.com/file/d/1atzhtv-kRegnabROGZs6olxuVONiRQKI/view?usp=sharing) | ![PixArt-α](https://drive.google.com/uc?export=view&id=1LckcGgmkpk4jL23u6eIRC_DC-DJiGw5S)
| [FLUX](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/flux_dyt.yaml) | [google drive](https://drive.google.com/file/d/1c0aWOJ4zfrvJ1kqDxfCCExRFxKBc34cZ/view?usp=sharing) | ![Flux](https://drive.google.com/uc?export=view&id=1Jn2mjodaK25k-CECMFvJ7_MK8s5us6xu)


After training the PixArt-α network for 20k steps, the PixArt-α model pipeline is able to generate samples like the below:

![PixArt-α](https://drive.google.com/uc?export=view&id=1LckcGgmkpk4jL23u6eIRC_DC-DJiGw5S)

The prompts we used for generation above were:

<pre>
2 2 7 9 1 zero 4 six 
4 7 5 four 9 eight 7 four 
6 four 6 zero 3 four seven 2 
six eight 1 six five three seven five 
zero 1 seven 3 six nine six seven 
eight 6 eight 7 five 1 7 one 
0 0 6 9 seven 0 1 four 
three seven six six 7 eight 1 three
</pre>

## Other Resources

There was no code or repo released with this paper.
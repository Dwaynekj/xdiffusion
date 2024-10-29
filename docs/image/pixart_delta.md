# PixArt-δ: Fast and Controllable Image Generation with Latent Consistency Models

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to build the PixArt-δ diffusion model from [PIXART-δ: Fast and Controllable Image Generation with Latent Consistency Models](https://arxiv.org/abs/2401.05252). PixArt-δ extends the PixArt-α diffusion model from [PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://arxiv.org/abs/2310.00426) (see [PixArt-α](https://github.com/swookey-thinky/xdiffusion/tree/main/docs/image/pixart_alpha.md)) by adding text alignment using cross attention at each transformer block.

## Introduction

PixArt-δ improves upon PixArt-α by adding a [ControlNet](https://arxiv.org/abs/2302.05543) architecture for Transformer backbones, and introduces a [latent consistency model](https://arxiv.org/abs/2310.04378) for inference.


In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Setting Up Your Environment

Follow the instructions from [Requirements For All Lessons](https://github.com/swookey-thinky/xdiffusion?tab=readme-ov-file#requirements) to set up your environment.


## Configuration File

The configuration file is located in [DiT](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/pixart_alpha.yaml).

## Running the Lesson

PixArt-α is a three stage training process, and we implement the first two stages here. First, we train a class conditional diffusion model without text captions, to create a strong pixel generation baseline.

Run the training script at `train_mnist.py` like:

```
> python training/image/train.py --config_path "configs/image/mnist/pixart_alpha_class_conditional.yaml" --num_training_steps 30000 --dataset_name "image/mnist"
```

Generated samples will be saved to `output/image/mnist/pixart_alpha_class_conditional`.

To train the second stage, run the same training script passing in the saved checkpoint from the first stage:

```
> python training/image/mnist/train.py --config_path "configs/image/mnist/pixart_alpha.yaml" --num_training_steps 30000 --load_model_weights_from_checkpoint "output/image/mnist/pixart_alpha_class_conditional/diffusion-30000.pt"
```

Generated samples and model checkpoints will be saved to `output/image/mnist/pixart_alpha`.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python sampling/image/mnist/sample.py --config_path configs/image/mnist/pixart_alpha.yaml --num_samples 8 --checkpoint output/image/mnist/pixart_alpha/diffusion-30000.pt
```

Output will be saved to the `output/image/mnist/sample/pixart_alpha` directory.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/pixart_alpha.yaml) | https://drive.google.com/file/d/19lIOUKg-GCg8g4PC7VXQ54fG4sS1ZsMB/view?usp=sharing | ![Pixart-Alpha](https://drive.google.com/uc?export=view&id=17hrD-Zxreb7XNpETWE4MdfVeqs1fnQXu)


After training the prior network for 30k steps, the full PixArt-α pipeline is able to generate samples like the below:

![Pixart-Alpha](https://drive.google.com/uc?export=view&id=17hrD-Zxreb7XNpETWE4MdfVeqs1fnQXu)

The prompts we used for generation above were:

<pre>
one two 1 2 one five 0 one 
2 0 5 two four zero eight five 
eight 0 8 five three nine 2 1 
0 6 four seven five 1 4 0 
0 seven four 1 9 zero one three 
3 nine zero 8 nine two 5 7 
zero 8 0 2 four 9 6 eight 
9 4 seven two eight eight one one 
</pre>

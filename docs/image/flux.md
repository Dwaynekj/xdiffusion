# FLUX

[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![pytorch](https://img.shields.io/badge/PyTorch-2.0.0-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)

In this lesson we are going to build the Flux diffusion model from Black Forest Labs. There is no technical report yet, but they released their model and inference code [here](https://github.com/black-forest-labs/flux) and we can infer a lot about how the model is built.

## Introduction 
Flux is a very cool model that combines a lot of recent research about image diffusion models. It extends the diffusion transformer network from [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748) with parallel transformer blocks from [Scaling Vision Transformers to 22 Billion Parameters](https://arxiv.org/abs/2302.05442). It uses the rectified flow formulation and MMDiT blocks from [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) which builds on flow matching from [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747). It also uses an improved positional embedding in the transformer blocks call rotary positional embedding, from [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864). The dev and schnell models that were released are alo guidance-distilled, using the approach from [Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042). This also sounds a lot like what Stable Diffusion 3 is built upon (minus the rotary position embedding and distillation), which makes sense since the authors were all key contributors to the stable diffusion series of models.

One of the interesting changes between SD3 and Flux here is that Flux includes both the dual-stream MMDiT blocks, as well as single stream blocks similar to DiT with the text tokens concatenated to the image tokens. They use twice the number of single stream blocks as they do double stream blocks, so curious to see the ablation study and rationale behind that design decision here.

In our implementation here, we train in pixel space rather than the latent space of the original model, and we scale down the transformer network to ~50m parameters from the original models 12b parameters. We save guidance-distillation for another lesson. We successfully trained this model on a T4 instance with a batch size of 64.

In this repository, we will be working with the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset because it is simple and can be trained in real time with minimal GPU power and memory. The main difference between MNIST and other datasets is the single channel of the imagery, versus 3 channels in most other datasets. We will make sure that the models we build can easily accomodate 1- or 3-channel data, so that you can test the models we build on other datasets.

## Configuration File

The configuration file is located in [Flux](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/flux.yaml).

## Training

To train the flux model, use:

```
> python training/image/mnist/train.py --config_path configs/image/mnist/flux.yaml
```

We successfully tested training on a single T4 instance (16GB VRAM) using a batch size of 64.

## Sampling

To sample from a pretrained checkpoint, you can run:

```
> python sampling/image/mnist/sample.py --config_path configs/image/mnist/flux.yaml --num_samples 8 --checkpoint output/image/mnist/flux/diffusion-10000.pt
```

Output will be saved to the `output/image/mnist/sample/flux` directory.

## Results and Checkpoints

| Config | Checkpoint | Results
| ------ | ---------- | -------
| [config](https://github.com/swookey-thinky/xdiffusion/blob/main/configs/image/mnist/flux.yaml) | [google drive](https://drive.google.com/file/d/1atzhtv-kRegnabROGZs6olxuVONiRQKI/view?usp=sharing) | ![Flux](https://drive.google.com/uc?export=view&id=1_r8poe1SJxf8UtT4mmQaTT378m26hD-F)


After training the network for 30k steps, the flux model pipeline is able to generate samples like the below:

![Flux](https://drive.google.com/uc?export=view&id=1_r8poe1SJxf8UtT4mmQaTT378m26hD-F)

The prompts we used for generation above were:

<pre>
3 5 six 5 six 9 six four 
5 zero eight one six 9 1 three 
two 2 8 8 zero six seven 5 
4 6 0 3 three 6 six four 
one 5 nine 2 5 6 2 three 
six four 0 4 one 8 eight five 
0 8 5 two four nine seven 1 
seven four four 2 eight 9 8 4 
</pre>

## Other Resources

The others released their inference code and model checkpoints [here](https://github.com/black-forest-labs/flux).
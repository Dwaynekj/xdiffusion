# xdiffusion
A unified media (Image, Video, Audio, Text) diffusion repository, for education and learning.

## Requirements

This package built using PyTorch and written in Python 3. To setup an environment to run all of the lessons, we suggest using conda or venv:

```
> python3 -m venv xdiffusion_env
> source xdiffusion_env/bin/activate
> pip install --upgrade pip
> pip install -r requirements.txt
```

All lessons are designed to be run from the root of the repository, and you should set your python path to include the repository root:

```
> export PYTHONPATH=$(pwd)
```

If you have issues with PyTorch and different CUDA versions on your instance, make sure to install the correct version of PyTorch for the CUDA version on your machine. For example, if you have CUDA 11.8 installed, you can install PyTorch using:

```
> pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Image Diffusion

### Training Datasets

### Image Models
The following is a list of the supported image models, their current results, and a link to their configuration files and documentation.

| Date  | Name  | Paper | Config | Results | Instructions
| :---- | :---- | ----- | ------ | ----- | -----

## Video Diffusion

### Training Datasets

### Video Diffusion Models
The following is a list of the supported image models, their current results, and a link to their configuration files and documentation.

| Date  | Name  | Paper | Config | Results | Instructions
| :---- | :---- | ----- | ------ | ----- | -----

## TODO

- [ ] Port the image diffusion repository
- [ ] Port the video diffusion respository
- [ ] Unify all of the different attention mechanisms under Transformer based (B, L, D) and pixel based (B, C, H, W).
"""Utility functions for working with the lesson."""

from einops import rearrange
from functools import partial
import importlib
import math
import numpy as np
import os
from packaging.version import Version, parse
import soundfile as sf
import sys
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import yaml

if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


class DotConfig:
    """Helper class to allow "." access to dictionaries."""

    def __init__(self, cfg):
        self._cfg = cfg

    def __getattr__(self, k) -> Any:
        v = self._cfg[k]
        if isinstance(v, dict):
            return DotConfig(v)
        return v

    def __getitem__(self, k) -> Any:
        return self.__getattr__(k)

    def __contains__(self, k) -> bool:
        try:
            v = self._cfg[k]
            return True
        except KeyError:
            return False

    def to_dict(self):
        return self._cfg


def load_yaml(yaml_path: str) -> DotConfig:
    """Loads a YAML configuration file."""
    with open(yaml_path, "r") as fp:
        return DotConfig(yaml.load(fp, yaml.CLoader))


def normalize_to_neg_one_to_one(img):
    """Converts tensors from (0,1) to (-1,1)."""
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    """Converts tensors from (-1,1) to (0,1)."""
    return (torch.clamp(t, -1.0, 1.0) + 1) * 0.5


def extract(a, t, x_shape):
    """Helper function to extract the values from a up until time t.

    This function is used to extract ranges of different constants up to the
    given time. The timestep  is a batched timestep of shape (B,) and dtype=torch.int32.
    This will gather the values of a at indices (timesteps) t - shape (B,) - and then
    reshape the (B,) output to match x_shape, appending dimensions of length 1. So if
    x_shape has shape (B,C,H,W), then the output will be of shape (B,1,1,1).
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps: int):
    """Linear beta schedule, proposed in original ddpm paper."""
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, max_beta: float = 0.999):
    """Cosine beta schedule, proposed in Improved DDPM."""
    alpha_bar = lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(timesteps):
        t1 = i / timesteps
        t2 = (i + 1) / timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.from_numpy(np.array(betas))


def normal_kl(mean1, logvar1, mean2, logvar2):
    """Computes the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.

    Original implementation:
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/losses.py#L12
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (
        1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    Args:
        x: Tensor batch of target images. It is assumed that this was uint8 values,
            rescaled to the range [-1, 1].
        means: Tensor batch of the Gaussian mean.
        log_scales: Tensor batch of the Gaussian log stddev.

    Returns:
        A tensor batch of log probabilities (in nats), of the same shape as x.
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(
            x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))
        ),
    )
    assert log_probs.shape == x.shape
    return log_probs


def cycle(dataloader: DataLoader):
    """Cycles through the dataloader class forever.

    Useful for when you want to cycle through a DataLoader for
    a finite number of timesteps.
    """
    while True:
        for data in dataloader:
            yield data


T = TypeVar("T", bound=torch.nn.Module)


def freeze(model: T) -> T:
    """Freeze the parameters of a model."""
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze(model: T) -> T:
    """Unfreeze the parameters of a model."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def instantiate_from_config(config, use_config_struct: bool = False) -> Any:
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if use_config_struct:
        return get_obj_from_str(config["target"])(config["params"])
    else:
        if "instantiate_with_config_struct" in config:
            if config["instantiate_with_config_struct"]:
                return get_obj_from_str(config["target"])(DotConfig(config["params"]))
            else:
                return get_obj_from_str(config["target"])(
                    **config.get("params", dict())
                )
        else:
            return get_obj_from_str(config["target"])(**config.get("params", dict()))


def instantiate_partial_from_config(config, use_config_struct: bool = False) -> Any:
    if not "target" in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    if use_config_struct:
        return partial(get_obj_from_str(config["target"]), config["params"])
    else:
        return partial(
            get_obj_from_str(config["target"]), **config.get("params", dict())
        )


def type_from_config(config) -> Type:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])


def kwargs_from_config(config) -> Dict:
    if not "params" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return config.get("params", dict())


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def fix_torchinfo_for_str():
    import sys
    import torchinfo

    def get_total_memory_used(
        data: torchinfo.torchinfo.CORRECTED_INPUT_DATA_TYPE,
    ) -> int:
        """Calculates the total memory of all tensors stored in data."""
        result = torchinfo.torchinfo.traverse_input_data(
            data,
            action_fn=lambda data: sys.getsizeof(
                data.untyped_storage()
                if hasattr(data, "untyped_storage")
                else data.storage()
            ),
            aggregate_fn=(
                # We don't need the dictionary keys in this case
                # if the data is not integer, assume the above action_fn is not applied for some reason
                (
                    lambda data: (
                        lambda d: (
                            sum(d.values())
                            if isinstance(d, torchinfo.torchinfo.Mapping)
                            else sys.getsizeof(d)
                        )
                    )
                )
                if (
                    isinstance(data, torchinfo.torchinfo.Mapping)
                    or not isinstance(data, int)
                )
                else sum
            ),
        )
        return torchinfo.torchinfo.cast(int, result)

    torchinfo.torchinfo.get_total_memory_used = get_total_memory_used


def get_constant_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, num_warmup_steps: int, last_epoch: int = -1
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    print(
        f"Creating constant learning rate schedule with {num_warmup_steps} warmup steps."
    )

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=last_epoch
    )


def broadcast_from_left(x, shape):
    assert len(shape) >= x.ndim
    return torch.broadcast_to(x.reshape(x.shape + (1,) * (len(shape) - x.ndim)), shape)


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def log1mexp(x):
    """Accurate computation of log(1 - exp(-x)) for x > 0."""
    # From James Townsend's PixelCNN++ code
    # Method from
    # https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    return torch.where(
        x > np.log(2), torch.log1p(-torch.exp(-x)), torch.log(-torch.expm1(-x))
    )


def dynamic_thresholding(x, p=0.995, c=1.7):
    """
    Dynamic thresholding, a diffusion sampling technique from Imagen (https://arxiv.org/abs/2205.11487)
    to leverage high guidance weights and generating more photorealistic and detailed images
    than previously was possible based on x.clamp(-1, 1) vanilla clipping or static thresholding

    p — percentile determine relative value for clipping threshold for dynamic compression,
        helps prevent oversaturation recommend values [0.96 — 0.99]

    c — absolute hard clipping of value for clipping threshold for dynamic compression,
        helps prevent undersaturation and low contrast issues; recommend values [1.5 — 2.]
    """
    x_shapes = x.shape
    s = torch.quantile(x.abs().reshape(x_shapes[0], -1), p, dim=-1)
    s = torch.clamp(s, min=1, max=c)
    x_compressed = torch.clip(x.reshape(x_shapes[0], -1).T, -s, s) / s
    x_compressed = x_compressed.T.reshape(x_shapes)
    return x_compressed


def largest_perfect_square(n):
    """
    Find the largest perfect square less than or equal to a given number n.

    Parameters:
        n (int): The input number.

    Returns:
        int: The largest perfect square less than or equal to n.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")

    # Calculate the integer part of the square root of n
    sqrt_n = int(math.sqrt(n))

    # Return the square of the result
    return sqrt_n**2


def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    B = tensor.shape[0]

    # Make sure the batch size has roots
    num_samples = largest_perfect_square(B)
    tensor = tensor[:num_samples, ...]

    # Convert the tensor of (B, C, F, H, W) to a grid of (C, F, H*sqrt(B), W*sqrt(b))
    images_grid = rearrange(
        tensor, "(i j) c f h w -> c f (i h) (j w)", i=int(math.sqrt(tensor.shape[0]))
    )
    images = map(v2.ToPILImage(), images_grid.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(
        path,
        save_all=True,
        append_images=rest_imgs,
        duration=duration,
        loop=loop,
        optimize=optimize,
    )
    return images


def save_mel_spectrogram_audio(mel_spec: torch.Tensor, path_spec: str):
    # Convert to wav
    from xdiffusion.layers.audio import mel_to_wav

    batch_mel_spec = mel_spec.cpu().numpy()
    for idx in tqdm(range(batch_mel_spec.shape[0]), leave=False, desc="Saving"):
        wav = mel_to_wav(mel_spec=batch_mel_spec[idx][0], sample_rate=16000)
        sf.write(path_spec.format(idx=idx), wav, 16000)


USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})


def _is_package_available(pkg_name: str):
    pkg_exists = importlib.util.find_spec(pkg_name) is not None
    pkg_version = "N/A"

    if pkg_exists:
        try:
            pkg_version = importlib_metadata.version(pkg_name)
        except (ImportError, importlib_metadata.PackageNotFoundError):
            pkg_exists = False

    return pkg_exists, pkg_version


if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
    _torch_available, _torch_version = _is_package_available("torch")
else:
    _torch_available = False


def is_torch_available():
    return _torch_available


def is_torch_version(operation: str, version: str):
    """
    Compares the current PyTorch version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of PyTorch
    """
    return compare_versions(parse(_torch_version), operation, version)


import operator as op

STR_OPERATION_TO_FUNC = {
    ">": op.gt,
    ">=": op.ge,
    "==": op.eq,
    "!=": op.ne,
    "<=": op.le,
    "<": op.lt,
}


def compare_versions(
    library_or_version: Union[str, Version], operation: str, requirement_version: str
):
    """
    Compares a library version to some requirement using a given operation.

    Args:
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(
            f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}"
        )
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = (
            generator.device.type
            if not isinstance(generator, list)
            else generator[0].device.type
        )
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(
                f"Cannot generate a {device} tensor from a generator of type {gen_device_type}."
            )

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(
                shape,
                generator=generator[i],
                device=rand_device,
                dtype=dtype,
                layout=layout,
            )
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(
            shape, generator=generator, device=rand_device, dtype=dtype, layout=layout
        ).to(device)

    return latents

import math
from functools import partial
import torch
import torch.nn as nn
from typing import Optional

from xdiffusion.layers.activation import Activation, GEGLU, GELU, ApproximateGELU


def get_initializer(name, activation=None):
    if activation in [None, "id", "identity", "linear", "modrelu"]:
        nonlinearity = "linear"
    elif activation in ["relu", "tanh", "sigmoid"]:
        nonlinearity = activation
    elif activation in ["gelu", "swish", "silu"]:
        nonlinearity = "relu"  # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(
            f"get_initializer: activation {activation} not supported"
        )

    if name == "uniform":
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == "normal":
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == "xavier":
        initializer = torch.nn.init.xavier_normal_
    elif name == "zero":
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == "one":
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(
            f"get_initializer: initializer type {name} not supported"
        )

    return initializer


def LinearActivation(
    d_input,
    d_output,
    bias=True,
    zero_bias_init=False,
    transposed=False,
    initializer=None,
    activation=None,
    activate=False,  # Apply activation as part of this module
    weight_norm=False,
    **kwargs,
):
    """Returns a linear nn.Module with control over axes order, initialization, and activation."""

    # Construct core module
    # linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation is not None and activation.startswith("glu"):
        d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    # Initialize weight
    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    # Initialize bias
    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    # Weight norm
    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, d_output, dim=1 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


class TransposedLinear(nn.Module):
    """Linear module on the second-to-last dimension.

    Assumes shape (B, D, L), where L can be 1 or more axis.
    """

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        # nn.init.kaiming_uniform_(self.weight, nonlinearity='linear') # should be equivalent
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # nn.Linear default init

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
            setattr(self.bias, "_optim", {"weight_decay": 0.0})
        else:
            self.bias = 0.0

    def forward(self, x):
        num_axis = len(x.shape[2:])  # num_axis in L, for broadcasting bias
        y = torch.einsum("b u ..., v u -> b v ...", x, self.weight) + self.bias.view(
            -1, *[1] * num_axis
        )
        return y


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = nn.Linear

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        else:
            raise ValueError(f"Unsupported activation function: {activation_fn}")

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(linear_cls(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        compatible_cls = (GEGLU,)
        for module in self.net:
            if isinstance(module, compatible_cls):
                hidden_states = module(hidden_states, scale)
            else:
                hidden_states = module(hidden_states)
        return hidden_states

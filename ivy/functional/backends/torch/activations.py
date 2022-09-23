"""Collection of PyTorch activation functions, wrapped to fit Ivy syntax and
signature.
"""
from typing import Optional, Union

# global
import numpy as np
import torch
import torch.nn

# local
import ivy


def relu(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.relu(x)


relu.unsupported_dtypes = ("float16",)


def leaky_relu(
    x: torch.Tensor,
    /,
    *,
    alpha: float = 0.2,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.nn.functional.leaky_relu(x, alpha)


leaky_relu.unsupported_dtypes = ("float16",)


def gelu(
    x: torch.Tensor, /, *, approximate: bool = True, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if approximate:
        return (
            0.5 * x * (1 + torch.tanh(((2 / np.pi) ** 0.5) * (x + 0.044715 * x**3)))
        )
    return torch.nn.functional.gelu(x)


gelu.unsupported_dtypes = ("float16",)


def sigmoid(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not ivy.is_array(x):
        x = torch.tensor(x)
    return torch.sigmoid(x, out=out)


sigmoid.unsupported_dtypes = ("float16",)
sigmoid.support_native_out = True


def softmax(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if axis is None:
        axis = -1
    return torch.nn.functional.softmax(x, axis)


softmax.unsupported_dtypes = ("float16",)


def softplus(
    x: torch.Tensor,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    kwargs = {
        k: v for k, v in {"beta": beta, "threshold": threshold}.items() if v is not None
    }
    return torch.nn.functional.softplus(x, **kwargs)


softplus.unsupported_dtypes = ("float16", "bfloat16")


def log_softmax(
    x: torch.Tensor,
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[torch.Tensor] = None,
):
    return torch.nn.functional.log_softmax(x, axis)


log_softmax.unsupported_dtypes = ("float16", "bfloat16")

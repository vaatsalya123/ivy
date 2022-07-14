"""Collection of PyTorch random functions, wrapped to fit Ivy syntax and signature."""

# global
import torch
from typing import Optional, List, Union, Sequence

# local
import ivy
from ivy.functional.ivy.device import default_device


# Extra #
# ------#


def random_uniform(
    low: float = 0.0,
    high: float = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    dtype=None,
    *,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rand_range = high - low
    if shape is None:
        shape = []
    return (
        torch.rand(shape, device=default_device(device), dtype=dtype, out=out)
        * rand_range
        + low
    )


def random_normal(
    mean: float = 0.0,
    std: float = 1.0,
    shape: Optional[Union[ivy.NativeShape, Sequence[int]]] = None,
    *,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if shape is None:
        true_shape: List[int] = []
    else:
        true_shape: List[int] = shape
    mean = mean.item() if isinstance(mean, torch.Tensor) else mean
    std = std.item() if isinstance(std, torch.Tensor) else std
    return torch.normal(mean, std, true_shape, device=default_device(device), out=out)


def multinomial(
    population_size: int,
    num_samples: int,
    batch_size: int = 1,
    probs: Optional[torch.Tensor] = None,
    replace: bool = True,
    *,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if probs is None:
        probs = (
            torch.ones(
                (
                    batch_size,
                    population_size,
                )
            )
            / population_size
        )
    return torch.multinomial(probs.float(), num_samples, replace, out=out).to(
        default_device(device)
    )


def randint(
    low: int,
    high: int,
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    device: torch.device,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.randint(low, high, shape, out=out, device=default_device(device))


def seed(seed_value: int = 0) -> None:
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    return


def shuffle(x: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    batch_size = x.shape[0]
    return torch.index_select(x, 0, torch.randperm(batch_size, out=out), out=out)

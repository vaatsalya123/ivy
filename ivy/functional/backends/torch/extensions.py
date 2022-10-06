from typing import Optional
import ivy
from ivy.functional.ivy.extensions import (
    _verify_coo_components,
    _verify_csr_components,
    _is_data_not_indices_values_and_shape,
    _is_coo_not_csr,
)
from ivy.functional.backends.torch.elementwise import _cast_for_unary_op
import torch
from math import sin, pi


def is_native_sparse_array(x):
    return x.layout in [torch.sparse_coo, torch.sparse_csr]


def native_sparse_array(
    data=None,
    *,
    coo_indices=None,
    csr_crow_indices=None,
    csr_col_indices=None,
    values=None,
    dense_shape=None,
):
    if _is_data_not_indices_values_and_shape(
        data, coo_indices, csr_crow_indices, csr_col_indices, values, dense_shape
    ):
        ivy.assertions.check_true(
            ivy.is_native_sparse_array(data), message="not a sparse array"
        )
        return data
    elif _is_coo_not_csr(
        coo_indices, csr_crow_indices, csr_col_indices, values, dense_shape
    ):
        _verify_coo_components(
            indices=coo_indices, values=values, dense_shape=dense_shape
        )
        return torch.sparse_coo_tensor(
            indices=coo_indices, values=values, size=dense_shape
        )
    else:
        _verify_csr_components(
            crow_indices=csr_crow_indices,
            col_indices=csr_col_indices,
            values=values,
            dense_shape=dense_shape,
        )
        return torch.sparse_csr_tensor(
            crow_indices=csr_crow_indices,
            col_indices=csr_col_indices,
            values=values,
            size=dense_shape,
        )


def native_sparse_array_to_indices_values_and_shape(x):
    if x.layout == torch.sparse_coo:
        x = x.coalesce()
        return x.indices(), x.values(), x.size()
    elif x.layout == torch.sparse_csr:
        return [x.crow_indices(), x.col_indices()], x.values(), x.size()
    raise ivy.exceptions.IvyException("not a sparse COO/CSR Tensor")


def sinc(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.sinc(x, out=out)


sinc.support_native_out = True
sinc.unsupported_dtypes = ("float16",)


def vorbis_window(
    window_length: torch.tensor,
    *,
    dtype:Optional[torch.dtype] = torch.float32,
    out: Optional[torch.tensor] = None
) -> torch.tensor:
    return torch.tensor([
        round(sin((pi/2)*(sin(pi*(i)/(window_length*2))**2)), 8)
        for i in range(1, window_length*2)[0::2]
    ], dtype=dtype)


vorbis_window.support_native_out = False
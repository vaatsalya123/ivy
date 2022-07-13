# global
import jax.numpy as jnp
from typing import Union, Optional, List, Sequence
import jaxlib.xla_extension
from jax.dlpack import from_dlpack as jax_from_dlpack

# local
import ivy
from ivy import as_native_dtype
from ivy.functional.backends.jax import JaxArray
from ivy.functional.backends.jax.device import _to_device
from ivy.functional.ivy.device import default_device
from ivy.functional.ivy import default_dtype


# Array API Standard #
# -------------------#


def arange(
    start,
    stop=None,
    step=1,
    *,
    dtype: jnp.dtype = None,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None
):
    if dtype:
        dtype = as_native_dtype(dtype)
    res = _to_device(jnp.arange(start, stop, step=step, dtype=dtype), device=device)
    if not dtype:
        if res.dtype == jnp.float64:
            return res.astype(jnp.float32)
        elif res.dtype == jnp.int64:
            return res.astype(jnp.int32)
    return res


def asarray(
    object_in,
    *,
    copy: Optional[bool] = None,
    dtype: jnp.dtype = None,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None
):
    if isinstance(object_in, ivy.NativeArray) and dtype != "bool":
        dtype = object_in.dtype
    elif (
        isinstance(object_in, (list, tuple, dict))
        and len(object_in) != 0
        and dtype is None
    ):
        dtype = default_dtype(item=object_in, as_native=True)
        if copy is True:
            return _to_device(
                jnp.array(object_in, dtype=dtype, copy=True), device=device
            )
        else:
            return _to_device(jnp.asarray(object_in, dtype=dtype), device=device)
    else:
        dtype = default_dtype(dtype, object_in)

    if copy is True:
        return _to_device(jnp.array(object_in, dtype=dtype, copy=True), device=device)
    else:
        return _to_device(jnp.asarray(object_in, dtype=dtype), device=device)


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return _to_device(
        jnp.empty(shape, as_native_dtype(default_dtype(dtype))), device=device
    )


def empty_like(
    x: JaxArray,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if dtype and str:
        dtype = jnp.dtype(dtype)
    else:
        dtype = x.dtype

    return _to_device(jnp.empty_like(x, dtype=dtype), device=device)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None
) -> JaxArray:
    dtype = as_native_dtype(default_dtype(dtype))
    device = default_device(device)
    return _to_device(jnp.eye(n_rows, n_cols, k, dtype), device=device)


# noinspection PyShadowingNames
def from_dlpack(
    x,
    *,
    out: Optional[JaxArray] = None
):
    return jax_from_dlpack(x)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float],
    *,
    dtype: jnp.dtype = None,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return _to_device(
        jnp.full(shape, fill_value, as_native_dtype(default_dtype(dtype, fill_value))),
        device=device,
    )


def full_like(
    x: JaxArray,
    fill_value: Union[int, float],
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if dtype and str:
        dtype = jnp.dtype(dtype)
    else:
        dtype = x.dtype

    return _to_device(
        jnp.full_like(
            x, fill_value, dtype=as_native_dtype(default_dtype(dtype, fill_value))
        ),
        device=device,
    )


def linspace(
    start,
    stop,
    num,
    axis=None,
    endpoint=True,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None
):
    if axis is None:
        axis = -1
    ans = jnp.linspace(start, stop, num, endpoint, dtype=dtype, axis=axis)
    if dtype is None:
        ans = jnp.float32(ans)
    return _to_device(ans, device=device)


def meshgrid(*arrays: JaxArray, indexing: str = "xy") -> List[JaxArray]:
    return jnp.meshgrid(*arrays, indexing=indexing)


def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: Optional[Union[ivy.Dtype, jnp.dtype]] = None,
    device: Optional[Union[ivy.Device, jaxlib.xla_extension.Device]] = None,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return _to_device(
        jnp.ones(shape, as_native_dtype(default_dtype(dtype))), device=device
    )


def ones_like(
    x: JaxArray,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if dtype and str:
        dtype = jnp.dtype(dtype)
    else:
        dtype = x.dtype
    return _to_device(jnp.ones_like(x, dtype=dtype), device=device)


def tril(
    x: JaxArray,
    k: int = 0,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.tril(x, k)


def triu(
    x: JaxArray,
    k: int = 0,
    *,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.triu(x, k)


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None
) -> JaxArray:
    return _to_device(
        jnp.zeros(shape, dtype),
        device=device,
    )


def zeros_like(
    x: JaxArray,
    *,
    dtype: jnp.dtype,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None
) -> JaxArray:
    if not dtype:
        dtype = x.dtype
    return _to_device(jnp.zeros_like(x, dtype=dtype), device=device)


# Extra #
# ------#


array = asarray


def logspace(
    start,
    stop,
    num,
    base=10.0,
    axis=None,
    *,
    device: jaxlib.xla_extension.Device,
    out: Optional[JaxArray] = None
):
    if axis is None:
        axis = -1
    return _to_device(
        jnp.logspace(start, stop, num, base=base, axis=axis), device=device
    )

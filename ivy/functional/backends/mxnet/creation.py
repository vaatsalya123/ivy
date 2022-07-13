# global
import mxnet as mx
from typing import Union, List, Optional, Iterable, Sequence
from numbers import Number

# local
import ivy
from ivy import default_device, as_native_dtype, default_dtype, as_ivy_dtype
from ivy.functional.backends.mxnet import _mxnet_init_context
from ivy.functional.backends.mxnet import _1_dim_array_to_flat_array


# Array API Standard #
# -------------------#


def _linspace(start, stop, num, cont):
    if num == 1:
        return start
    start = mx.nd.array(start).reshape((1,)).astype("float32")
    stop = mx.nd.array(stop).reshape((1,)).astype("float32")
    n_m_1 = mx.nd.array(num - 1).reshape((1,)).astype("float32")
    increment = (stop - start) / n_m_1
    increment_tiled = mx.nd.tile(increment, num - 1)
    increments = increment_tiled * mx.nd.array(
        mx.nd.np.linspace(1, num - 1, num - 1).tolist(), ctx=cont
    )
    ret = mx.nd.concat(start, start + increments, dim=0)
    return ret


def arange(stop, start=0, step=1, dtype=None, device=None):
    cont = _mxnet_init_context(default_device(device))
    stop = stop if isinstance(stop, Number) else stop.asscalar()
    start = start if isinstance(start, Number) else start.asscalar()
    step = step if isinstance(step, Number) else step.asscalar()
    return mx.nd.arange(start, stop, ctx=cont, step=step, dtype=dtype)


def asarray(
    object_in,
    dtype: Optional[Union[ivy.Dtype, type]] = None,
    device: Optional[Union[ivy.Device, mx.context.Context]] = None,
    copy: Optional[bool] = None,
):
    # mxnet don't have asarray implementation, haven't properly tested
    cont = _mxnet_init_context(default_device(device))
    if copy is None:
        copy = False
    if copy:
        if dtype is None and isinstance(object_in, mx.nd.NDArray):
            return mx.nd.array(object_in, cont).as_in_context(cont)
        if dtype is None and not isinstance(object_in, mx.nd.NDArray):
            return mx.nd.array(object_in, cont, dtype=default_dtype(dtype, object_in))
        else:
            dtype = as_ivy_dtype(default_dtype(dtype, object_in))
            return mx.nd.array(object_in, cont, dtype=default_dtype(dtype, object_in))
    else:
        if dtype is None and isinstance(object_in, mx.nd.NDArray):
            return object_in.as_in_context(cont)
        if dtype is None and not isinstance(object_in, mx.nd.NDArray):
            return mx.nd.array(object_in, cont, dtype=default_dtype(dtype, object_in))
        else:
            dtype = as_ivy_dtype(default_dtype(dtype, object_in))
            return mx.nd.array(object_in, cont, dtype=default_dtype(dtype, object_in))


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    dtype: Optional[Union[ivy.Dtype, type]] = None,
    device: Optional[Union[ivy.Device, mx.context.Context]] = None,
) -> mx.nd.NDArray:
    cont = _mxnet_init_context(default_device(device))
    return mx.nd.empty(shape, as_native_dtype(default_dtype(dtype)), cont)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    dtype: Optional[Union[ivy.Dtype, type]] = None,
    device: Optional[Union[ivy.Device, mx.context.Context]] = None,
) -> mx.nd.NDArray:
    cont = _mxnet_init_context(default_device(device))
    return mx.nd.eye(n_rows, n_cols, k, ctx=cont).astype(dtype)


# noinspection PyUnresolvedReferences
def from_dlpack(x):
    return mx.nd.from_dlpack(x)


def full(
    shape: Union[ivy.NativeShape, Sequence[int]], fill_value, dtype=None, device=None
):
    shape = ivy.shape_to_tuple(shape)
    cont = _mxnet_init_context(default_device(device))
    if len(shape) == 0 or 0 in shape:
        return _1_dim_array_to_flat_array(
            mx.nd.full(
                (1,),
                fill_value,
                cont,
                as_native_dtype(default_dtype(dtype, fill_value)),
            )
        )
    return mx.nd.full(
        shape, fill_value, cont, as_native_dtype(default_dtype(dtype, fill_value))
    )


def linspace(start, stop, num, axis=None, device=None):
    cont = _mxnet_init_context(default_device(device))
    num = num.asnumpy()[0] if isinstance(num, mx.nd.NDArray) else num
    start_is_array = isinstance(start, mx.nd.NDArray)
    stop_is_array = isinstance(stop, mx.nd.NDArray)
    start_shape = []
    if start_is_array:
        start_shape = list(start.shape)
        start = start.reshape((-1,))
    if stop_is_array:
        start_shape = list(stop.shape)
        stop = stop.reshape((-1,))
    if start_is_array and stop_is_array:
        res = [_linspace(strt, stp, num, cont) for strt, stp in zip(start, stop)]
    elif start_is_array and not stop_is_array:
        res = [_linspace(strt, stop, num, cont) for strt in start]
    elif not start_is_array and stop_is_array:
        res = [_linspace(start, stp, num, cont) for stp in stop]
    else:
        return _linspace(start, stop, num, cont)
    new_shape = start_shape + [num]
    res = mx.nd.concat(*res, dim=-1).reshape(new_shape)
    if axis is not None:
        res = mx.nd.swapaxes(res, axis, -1)
    return res


def meshgrid(*xs: mx.nd.NDArray, indexing: Optional[str] = "xy") -> List[mx.nd.NDArray]:
    # ToDo: implement this without reliance on NumPy backend
    xs_np = [x.as_np_ndarray() for x in xs]
    return [item.as_nd_ndarray() for item in mx.np.meshgrid(*xs_np, indexing=indexing)]


def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    dtype: Optional[Union[ivy.Dtype, type]] = None,
    device: Optional[Union[ivy.Device, mx.context.Context]] = None,
) -> mx.nd.NDArray:
    cont = _mxnet_init_context(default_device(device))
    shape = [shape] if shape is not isinstance(shape, Iterable) else shape
    if len(shape) == 0 or 0 in shape:
        return _1_dim_array_to_flat_array(mx.nd.ones((1,), ctx=cont).astype(dtype))
    return mx.nd.ones(shape, ctx=cont).astype(dtype)


def ones_like(
    x: mx.nd.NDArray,
    dtype: Optional[Union[ivy.Dtype, type]] = None,
    device: Optional[Union[ivy.Device, mx.context.Context]] = None,
) -> mx.nd.NDArray:
    if x.shape == ():
        return mx.nd.array(1.0, ctx=_mxnet_init_context(default_device(device)))
    mx_ones = mx.nd.ones_like(x, ctx=_mxnet_init_context(default_device(device)))
    return mx_ones if dtype is None else mx_ones.astype(dtype)


def tril(x: mx.nd.NDArray, k: int = 0) -> mx.nd.NDArray:
    return mx.np.tril(x, k)


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: type,
    device: mx.context.Context,
) -> mx.nd.NDArray:
    cont = _mxnet_init_context(device)
    if len(shape) == 0 or 0 in shape:
        return _1_dim_array_to_flat_array(mx.nd.zeros((1,), ctx=cont).astype(dtype))
    return mx.nd.zeros(shape, ctx=cont).astype(dtype)


def zeros_like(x, dtype=None, device=None):
    if x.shape == ():
        return mx.nd.array(0.0, ctx=_mxnet_init_context(default_device(device)))
    mx_zeros = mx.nd.zeros_like(x, ctx=_mxnet_init_context(default_device(device)))
    return mx_zeros if not dtype else mx_zeros.astype(dtype)


# Extra #
# ------#


array = asarray


def logspace(start, stop, num, base=10.0, axis=None, device=None):
    power_seq = linspace(start, stop, num, axis, default_device(device))
    return base**power_seq

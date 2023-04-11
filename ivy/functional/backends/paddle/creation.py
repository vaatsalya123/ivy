# global

from numbers import Number
from typing import Union, List, Optional, Sequence

import numpy as np
import paddle

# local
import ivy
from ivy.func_wrapper import (
    with_unsupported_dtypes,
    with_unsupported_device_and_dtypes,
    _get_first_array,
)
from ivy.functional.ivy.creation import (
    asarray_to_native_arrays_and_back,
    asarray_infer_device,
    asarray_handle_nestable,
    NestedSequence,
    SupportsBufferProtocol,
)
from . import backend_version
from paddle.fluid.libpaddle import Place
from ivy.functional.backends.paddle.device import to_device

# Array API Standard #
# -------------------#


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def arange(
    start: float,
    /,
    stop: Optional[float] = None,
    step: float = 1,
    *,
    dtype: Optional[Union[ivy.Dtype, paddle.dtype]] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if stop is None:
        stop = start
        start = 0
    if (step > 0 and start > stop) or (step < 0 and start < stop):
        if isinstance(stop, float):
            stop = float(start)
        else:
            stop = start
    if dtype is None:
        if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
            return to_device(
                paddle.arange(start, stop, step, dtype=paddle.int32), device
            )

        elif (
            isinstance(start, float)
            or isinstance(stop, float)
            or isinstance(step, float)
        ):
            return to_device(
                paddle.arange(start, stop, step, dtype=paddle.float32), device
            )

        else:
            return to_device(paddle.arange(start, stop, step), device)
    else:
        dtype = ivy.as_native_dtype(ivy.default_dtype(dtype=dtype))
        return to_device(paddle.arange(start, stop, step).cast(dtype), device)


def _stack_tensors(x, dtype):
    # TODO: change paddle.stack to ivy.stack
    if isinstance(x, (list, tuple)) and len(x) != 0 and isinstance(x[0], (list, tuple)):
        for i, item in enumerate(x):
            x[i] = _stack_tensors(item, dtype)
        x = ivy.stack(x)
    else:
        if isinstance(x, (list, tuple)):
            if isinstance(x[0], paddle.Tensor):
                x = ivy.stack([i for i in x])
            else:
                x = paddle.to_tensor(x, dtype=dtype)
    x.stop_gradient = False
    return x


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
@asarray_to_native_arrays_and_back
@asarray_infer_device
@asarray_handle_nestable
def asarray(
    obj: Union[
        paddle.Tensor,
        np.ndarray,
        bool,
        int,
        float,
        NestedSequence,
        SupportsBufferProtocol,
    ],
    /,
    *,
    copy: Optional[bool] = None,
    dtype: Optional[Union[ivy.Dtype, paddle.dtype]] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    # TODO: Implement device support

    if isinstance(obj, paddle.Tensor) and dtype is None:
        if copy is True:
            ret = obj.clone().detach()
            ret.stop_gradient = obj.stop_gradient
            return ret
        return obj

    elif isinstance(obj, (list, tuple, dict)) and len(obj) != 0:
        contain_tensor = False
        if isinstance(obj[0], (list, tuple)):
            first_tensor = _get_first_array(obj)
            if ivy.exists(first_tensor):
                contain_tensor = True
                dtype = first_tensor.dtype
        if dtype is None:
            dtype = ivy.default_dtype(item=obj, as_native=True)

        # if `obj` is a list of specifically tensors or
        # a multidimensional list which contains a tensor
        if isinstance(obj[0], paddle.Tensor) or contain_tensor:
            if copy is True:
                ret = ivy.stack([i for i in obj]).cast(dtype).clone().detach()
                ret.stop_gradient = obj[0].stop_gradient
                return ret
            else:
                return _stack_tensors(obj, dtype)

    elif isinstance(obj, np.ndarray) and dtype is None:
        dtype = ivy.as_native_dtype(ivy.as_ivy_dtype(obj.dtype.name))

    elif isinstance(obj, (Number, bool, complex)):
        if dtype is None:
            dtype = ivy.default_dtype(item=obj)
        with ivy.ArrayMode(False):
            return ivy.squeeze(paddle.to_tensor(obj, dtype=dtype), 0)

    else:
        dtype = ivy.as_native_dtype((ivy.default_dtype(dtype=dtype, item=obj)))

    if dtype == paddle.bfloat16 and isinstance(obj, np.ndarray):
        if copy is True:
            ret = paddle.to_tensor(obj.tolist(), dtype=dtype).clone().detach()
            return ret
        else:
            ret = paddle.to_tensor(obj.tolist(), dtype=dtype)
            return ret

    if copy is True:
        ret = paddle.to_tensor(obj, dtype=dtype).clone().detach()
        return ret
    else:
        if not ivy.is_native_array(obj):
            ret = paddle.to_tensor(obj, dtype=dtype)
        else:
            ret = obj.cast(dtype)
        return ret


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def empty(
    *size: Union[int, Sequence[int]],
    shape: Optional[ivy.NativeShape] = None,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if size and shape:
        raise TypeError("empty() got multiple values for argument 'shape'")
    if shape is None:
        shape = size[0] if isinstance(size[0], (tuple, list)) else size
    return to_device(paddle.empty(shape=shape).cast(dtype), device)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def empty_like(
    x: paddle.Tensor,
    /,
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return to_device(paddle.empty_like(x=x.cast("float32")).cast(dtype), device)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    /,
    *,
    k: int = 0,
    batch_shape: Optional[Union[int, Sequence[int]]] = None,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if n_cols is None:
        n_cols = n_rows
    i = paddle.eye(n_rows, n_cols)
    if batch_shape is None:
        return to_device(i.astype(dtype), device)
    reshape_dims = [1] * len(batch_shape) + [n_rows, n_cols]
    tile_dims = list(batch_shape) + [1, 1]
    i = ivy.broadcast_to(i, reshape_dims)
    return_mat = paddle.tile(i, tile_dims)
    return to_device(return_mat.astype(dtype), device)


def from_dlpack(x, /, *, out: Optional[paddle.Tensor] = None):
    x_d = paddle.utils.dlpack.to_dlpack(x)
    return paddle.utils.dlpack.from_dlpack(x_d)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float, bool],
    *,
    dtype: Optional[Union[ivy.Dtype, paddle.dtype]] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if dtype is None:
        dtype = ivy.default_dtype(item=fill_value)
    return to_device(
        paddle.full(shape=shape, fill_value=fill_value).cast(dtype), device
    )


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def full_like(
    x: paddle.Tensor,
    /,
    fill_value: Number,
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return full(shape=x.shape, fill_value=fill_value, dtype=dtype, device=device)


def _linspace_helper(start, stop, num, axis=None, *, dtype=None):
    num = num.detach().item() if isinstance(num, paddle.Tensor) else num
    start_is_array = isinstance(start, paddle.Tensor)
    stop_is_array = isinstance(stop, paddle.Tensor)
    linspace_method = paddle.linspace
    sos_shape = []
    if start_is_array:
        start_shape = start.shape
        sos_shape = start_shape
        if num == 1:
            if axis is not None:
                return ivy.expand_dims(start, axis=axis)
            else:
                return ivy.expand_dims(start, axis=-1)
        start = start.reshape((-1,))
        linspace_method = (
            _differentiable_linspace if not start.stop_gradient else paddle.linspace
        )
    if stop_is_array:
        stop_shape = stop.shape
        sos_shape = stop_shape
        if num == 1:
            return ivy.ones(stop_shape[:axis] + [1] + stop_shape[axis:]) * start
        stop = stop.reshape((-1,))
        linspace_method = (
            _differentiable_linspace if not stop.stop_gradient else paddle.linspace
        )
    if start_is_array and stop_is_array:
        if num < start.shape[0]:
            start = ivy.expand_dims(start, axis=-1)
            stop = ivy.expand_dims(stop, axis=-1)
            diff = stop - start
            inc = diff / (num - 1)
            res = [start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(stop)
        else:
            res = [
                linspace_method(strt, stp, num)
                for strt, stp in zip(ivy.unstack(start), ivy.unstack(stop))
            ]
    elif start_is_array and not stop_is_array:
        if num < start.shape[0]:
            start = ivy.expand_dims(start, axis=axis)
            diff = stop - start
            inc = diff / (num - 1)
            res = [start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(ivy.ones_like(start) * stop)
        else:
            res = [linspace_method(strt, stop, num) for strt in start]
    elif not start_is_array and stop_is_array:
        if num < stop.shape[0]:
            stop = ivy.expand_dims(stop, axis=-1)
            diff = stop - start
            inc = diff / (num - 1)
            res = [ivy.ones_like(stop) * start]
            res += [start + inc * i for i in range(1, num - 1)]
            res.append(stop)
        else:
            res = [linspace_method(start, stp, num) for stp in stop]
    else:
        return linspace_method(start, stop, num, dtype=dtype)
    res = ivy.concat(res, axis=-1).reshape(sos_shape + [num])
    if axis is not None:
        ndim = res.ndim
        perm = ivy.arange(0, ndim - 1).tolist()
        perm.insert(axis % (ndim + 1), ndim - 1)
        res = ivy.permute_dims(res, perm)
    return res


def _differentiable_linspace(start, stop, num, *, dtype=None):
    with ivy.ArrayMode(False):
        start = ivy.to_native(start)
        num = paddle.to_tensor(num, stop_gradient=False)
        if num == 1:
            return ivy.expand_dims(start, axis=0)
        n_m_1 = ivy.subtract(num, 1)
        increment = ivy.divide(ivy.subtract(stop, start), n_m_1)
        increment_tiled = ivy.repeat(increment, n_m_1)
        increments = ivy.multiply(
            increment_tiled,
            paddle.linspace(1, n_m_1, n_m_1.cast(paddle.int32), dtype=dtype),
        )
        if start.ndim == 0:
            start = ivy.expand_dims(start, axis=0)
        res = ivy.concat((start, ivy.add(start, increments)), axis=0)
    return res.cast(dtype)


def _slice_at_axis(sl, axis):
    return (slice(None),) * axis + (sl,) + (...,)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def linspace(
    start: Union[paddle.Tensor, float],
    stop: Union[paddle.Tensor, float],
    /,
    num: int,
    *,
    axis: Optional[int] = None,
    endpoint: bool = True,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if not isinstance(start, paddle.Tensor):
        start = paddle.to_tensor(start)

    if not isinstance(start, paddle.Tensor):
        start = paddle.to_tensor(stop)

    if not isinstance(start, paddle.Tensor):
        start = paddle.to_tensor(num)

    if axis is None:
        axis = -1
    if not endpoint:
        if dtype is not None:
            ans = _linspace_helper(start, stop, num + 1, axis, dtype=dtype)
        else:
            ans = _linspace_helper(start, stop, num + 1, axis)
        if axis < 0:
            axis += len(ans.shape)
        ans = ans[_slice_at_axis(slice(None, -1), axis)]
    else:
        if dtype is not None:
            ans = _linspace_helper(start, stop, num, axis, dtype=dtype)
        else:
            ans = _linspace_helper(start, stop, num, axis)
    if (
        endpoint
        and ans.shape[0] > 1
        and (not isinstance(start, paddle.Tensor))
        and (not isinstance(stop, paddle.Tensor))
    ):
        ans[-1] = stop
    if (
        ans.shape[0] >= 1
        and (not isinstance(start, paddle.Tensor))
        and (not isinstance(stop, paddle.Tensor))
        and ans[0] != start
    ):
        ans[0] = start
    if "int" in str(dtype) and paddle.is_floating_point(ans):
        ans = paddle.floor(ans)
    return to_device(ans.cast(dtype), device)


@with_unsupported_dtypes(
    {
        "2.4.2 and below": (
            "int8",
            "int16",
            "uint8",
            "uint16",
            "bfloat16",
            "float16",
            "complex64",
            "complex128",
            "bool",
        )
    },
    backend_version,
)
def meshgrid(
    *arrays: paddle.Tensor,
    sparse: bool = False,
    indexing: str = "xy",
) -> List[paddle.Tensor]:
    if not sparse:
        if indexing == "ij":
            return paddle.meshgrid(*arrays)
        elif indexing == "xy":
            return paddle.meshgrid(*arrays[::-1])[::-1]
        else:
            raise ValueError(f"indexing must be either 'ij' or 'xy', got {indexing}")

    sd = (1,) * len(arrays)
    res = [
        paddle.reshape(paddle.to_tensor(a), (sd[:i] + (-1,) + sd[i + 1 :]))
        for i, a in enumerate(arrays)
    ]

    if indexing == "xy" and len(arrays) > 1:
        res[0] = paddle.reshape(res[0], (1, -1) + sd[2:])
        res[1] = paddle.reshape(res[1], (-1, 1) + sd[2:])

    return res


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def ones(
    *size: Union[int, Sequence[int]],
    shape: Optional[ivy.NativeShape] = None,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if size and shape:
        raise TypeError("ones() got multiple values for argument 'shape'")
    if shape is None:
        shape = size[0] if isinstance(size[0], (tuple, list)) else size
    return to_device(paddle.ones(shape=shape).cast(dtype), device)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def ones_like(
    x: paddle.Tensor,
    /,
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    return to_device(paddle.ones_like(x=x.cast("float32")).cast(dtype), device)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "uint16",
                "bfloat16",
                "complex64",
                "complex128",
            )
        }
    },
    backend_version,
)
def tril(
    x: paddle.Tensor, /, *, k: int = 0, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.tril(x=x, diagonal=k)


@with_unsupported_device_and_dtypes(
    {
        "2.4.2 and below": {
            "cpu": (
                "int8",
                "int16",
                "uint8",
                "uint16",
                "bfloat16",
                "complex64",
                "complex128",
            )
        }
    },
    backend_version,
)
def triu(
    x: paddle.Tensor, /, *, k: int = 0, out: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    return paddle.triu(x=x, diagonal=k)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def zeros(
    *size: Union[int, Sequence[int]],
    shape: Optional[ivy.NativeShape] = None,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if size and shape:
        raise TypeError("zeros() got multiple values for argument 'shape'")
    if shape is None:
        shape = size[0] if isinstance(size[0], (tuple, list)) else size
    return to_device(paddle.zeros(shape=shape).cast(dtype), device)


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def zeros_like(
    x: paddle.Tensor,
    /,
    *,
    dtype: paddle.dtype,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if dtype is None:
        dtype = x.dtype
    return to_device(paddle.zeros_like(x=x.cast("float32")).cast(dtype), device)


# Extra #
# ------#


array = asarray


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def copy_array(
    x: paddle.Tensor,
    *,
    to_ivy_array: Optional[bool] = True,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    if to_ivy_array:
        return ivy.to_ivy(x.clone())
    return x.clone()


@with_unsupported_device_and_dtypes(
    {"2.4.2 and below": {"cpu": ("uint16", "bfloat16")}}, backend_version
)
def one_hot(
    indices: paddle.Tensor,
    depth: int,
    /,
    *,
    on_value: Optional[paddle.Tensor] = None,
    off_value: Optional[paddle.Tensor] = None,
    axis: Optional[int] = None,
    dtype: Optional[paddle.dtype] = None,
    device: Place,
    out: Optional[paddle.Tensor] = None,
) -> paddle.Tensor:
    on_none = on_value is None
    off_none = off_value is None
    if indices.ndim == 0:
        indices = indices.cast("int64").unsqueeze(0)
    if dtype is None:
        if on_none and off_none:
            dtype = paddle.float32
        else:
            if not on_none:
                dtype = paddle.to_tensor(on_value).dtype
            elif not off_none:
                dtype = paddle.to_tensor(off_value).dtype
    else:
        dtype = ivy.as_native_dtype(dtype)

    on_value = (
        paddle.to_tensor(1.0, dtype="float32")
        if on_none
        else paddle.to_tensor(on_value, dtype="float32")
    )
    off_value = (
        paddle.to_tensor(0.0, dtype="float32")
        if off_none
        else paddle.to_tensor(off_value, dtype="float32")
    )

    res = paddle.nn.functional.one_hot(indices.cast(paddle.int64), depth)

    if not on_none or not off_none:
        res = paddle.where(res == 1, on_value, off_value)

    if axis is not None:
        res = paddle.moveaxis(res, -1, axis)

    return to_device(res.cast(dtype), device)

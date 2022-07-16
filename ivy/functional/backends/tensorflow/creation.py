# global
import tensorflow as tf
from typing import Union, Sequence, List, Optional

# local
import ivy
from ivy import (
    as_native_dev,
    default_device,
    as_native_dtype,
    default_dtype,
    as_ivy_dtype,
)


# Array API Standard #
# -------------------#


def arange(start, stop=None, step=1, *, dtype: tf.DType = None, device: str):
    if stop is None:
        stop = start
        start = 0
    if (step > 0 and start > stop) or (step < 0 and start < stop):
        if isinstance(stop, float):
            stop = float(start)
        else:
            stop = start

    device = as_native_dev(default_device(device))
    with tf.device(device):

        if dtype is None:
            if (
                isinstance(start, int)
                and isinstance(stop, int)
                and isinstance(step, int)
            ):
                return tf.cast(
                    tf.range(start, stop, delta=step, dtype=tf.int64), tf.int32
                )
            else:
                return tf.range(start, stop, delta=step)
        else:
            dtype = as_native_dtype(default_dtype(dtype))
            if dtype in [tf.int8, tf.uint8, tf.int16, tf.uint16, tf.uint32, tf.uint64]:
                return tf.cast(tf.range(start, stop, delta=step, dtype=tf.int64), dtype)
            else:
                return tf.range(start, stop, delta=step, dtype=dtype)


def asarray(object_in, *, copy=None, dtype: tf.DType = None, device: str):
    device = default_device(device)
    with tf.device(as_native_dev(device)):
        if copy:
            if dtype is None and isinstance(object_in, tf.Tensor):
                return tf.identity(object_in)
            if dtype is None and not isinstance(object_in, tf.Tensor):
                try:
                    dtype = default_dtype(item=object_in, as_native=True)
                    tensor = tf.convert_to_tensor(object_in, dtype=dtype)
                except (TypeError, ValueError):
                    dtype = default_dtype(dtype, object_in, True)
                    tensor = tf.convert_to_tensor(
                        ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)),
                        dtype=dtype,
                    )
                return tf.identity(tf.cast(tensor, dtype))
            else:
                dtype = as_ivy_dtype(default_dtype(dtype, object_in))
                try:
                    tensor = tf.convert_to_tensor(object_in, dtype=dtype)
                except (TypeError, ValueError):
                    tensor = tf.convert_to_tensor(
                        ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)),
                        dtype=dtype,
                    )
                return tf.identity(tf.cast(tensor, dtype))
        else:
            if dtype is None and isinstance(object_in, tf.Tensor):
                return object_in
            if dtype is None and not isinstance(object_in, tf.Tensor):
                try:
                    return tf.convert_to_tensor(object_in)
                except (TypeError, ValueError):
                    dtype = as_ivy_dtype(default_dtype(dtype, object_in))
                    return tf.convert_to_tensor(
                        ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)),
                        dtype=dtype,
                    )
            else:
                dtype = as_ivy_dtype(default_dtype(dtype, object_in))
                try:
                    tensor = tf.convert_to_tensor(object_in, dtype=dtype)
                except (TypeError, ValueError):
                    tensor = tf.convert_to_tensor(
                        ivy.nested_map(object_in, lambda x: tf.cast(x, dtype)),
                        dtype=dtype,
                    )
                return tf.cast(tensor, dtype)


def empty(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: tf.DType,
    device: str,
) -> Union[tf.Tensor, tf.Variable]:
    device = default_device(device)
    with tf.device(as_native_dev(device)):
        return tf.experimental.numpy.empty(shape, as_native_dtype(default_dtype(dtype)))


def empty_like(
    x: Union[tf.Tensor, tf.Variable],
    *,
    dtype: tf.DType,
    device: str,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = tf.DType(dtype) if dtype is str else dtype
    device = default_device(device)
    with tf.device(as_native_dev(device)):
        return tf.experimental.numpy.empty_like(x, dtype=dtype)


def eye(
    n_rows: int,
    n_cols: Optional[int] = None,
    k: Optional[int] = 0,
    *,
    dtype: tf.DType,
    device: str,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = as_native_dtype(default_dtype(dtype))
    device = as_native_dev(default_device(device))
    with tf.device(device):
        if n_cols is None:
            n_cols = n_rows
        i = tf.eye(n_rows, n_cols, dtype=dtype)
        if k == 0:
            return i
        elif -n_rows < k < 0:
            return tf.concat([tf.zeros([-k, n_cols], dtype=dtype), i[: n_rows + k]], 0)
        elif 0 < k < n_cols:
            return tf.concat(
                [tf.zeros([n_rows, k], dtype=dtype), i[:, : n_cols - k]], 1
            )
        else:
            return tf.zeros([n_rows, n_cols], dtype=dtype)


# noinspection PyShadowingNames
def from_dlpack(x):
    return tf.experimental.dlpack.from_dlpack(x)


def _assert_fill_value_and_dtype_are_compatible(dtype, fill_value):
    assert (ivy.is_int_dtype(dtype) and isinstance(fill_value, int)) or (
        ivy.is_float_dtype(dtype) and isinstance(fill_value, float)
    ), "the fill_value and data type"


def full(
    shape: Union[ivy.NativeShape, Sequence[int]],
    fill_value: Union[int, float],
    *,
    dtype: Optional[Union[ivy.Dtype, tf.DType]] = None,
    device: str,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = ivy.default_dtype(dtype, item=fill_value, as_native=True)
    _assert_fill_value_and_dtype_are_compatible(dtype, fill_value)
    with tf.device(as_native_dev(default_device(device))):
        return tf.fill(
            shape,
            tf.constant(fill_value, dtype=dtype),
        )


def full_like(
    x: Union[tf.Tensor, tf.Variable],
    fill_value: Union[int, float],
    *,
    dtype: Optional[Union[ivy.Dtype, tf.DType]] = None,
    device: str,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = ivy.default_dtype(dtype, item=fill_value, as_native=True)
    _assert_fill_value_and_dtype_are_compatible(dtype, fill_value)
    device = as_native_dev(default_device(device))
    with tf.device(device):
        return tf.experimental.numpy.full_like(x, fill_value, dtype=dtype)


def linspace(
    start, stop, num, axis=None, endpoint=True, *, dtype: tf.DType, device: str
):
    if axis is None:
        axis = -1
    dtype = as_native_dtype(default_dtype(dtype))
    device = default_device(device)
    with tf.device(ivy.as_native_dev(device)):
        start = tf.constant(start, dtype=dtype)
        stop = tf.constant(stop, dtype=dtype)
        if not endpoint:
            ans = tf.linspace(start, stop, num + 1, axis=axis)[:-1]
        else:
            ans = tf.linspace(start, stop, num, axis=axis)
        ans = tf.cast(ans, dtype)
        return ans


def meshgrid(
    *arrays: Union[tf.Tensor, tf.Variable],
    indexing: str = "xy",
) -> List[Union[tf.Tensor, tf.Variable]]:
    return tf.meshgrid(*arrays, indexing=indexing)


def ones(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: tf.DType,
    device: str,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = as_native_dtype(default_dtype(dtype))
    device = as_native_dev(default_device(device))
    with tf.device(device):
        return tf.ones(shape, dtype)


def ones_like(
    x: Union[tf.Tensor, tf.Variable],
    *,
    dtype: tf.DType,
    device: str,
) -> Union[tf.Tensor, tf.Variable]:
    dtype = tf.DType(dtype) if dtype is str else dtype
    device = default_device(device)
    with tf.device(as_native_dev(device)):
        return tf.ones_like(x, dtype=dtype)


def tril(x: Union[tf.Tensor, tf.Variable], k: int = 0) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.tril(x, k)


def triu(x: Union[tf.Tensor, tf.Variable], k: int = 0) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.triu(x, k)


def zeros(
    shape: Union[ivy.NativeShape, Sequence[int]],
    *,
    dtype: tf.DType,
    device: str,
) -> Union[tf.Tensor, tf.Variable]:
    with tf.device(device):
        return tf.zeros(shape, dtype)


def zeros_like(
    x: Union[tf.Tensor, tf.Variable],
    *,
    dtype: tf.DType,
    device: str,
) -> Union[tf.Tensor, tf.Variable]:
    device = default_device(device)
    with tf.device(as_native_dev(device)):
        return tf.zeros_like(x, dtype=dtype)


# Extra #
# ------#


array = asarray


def logspace(start, stop, num, base=10.0, axis=None, *, device: str):
    power_seq = linspace(
        start, stop, num, axis, dtype=None, device=default_device(device)
    )
    return base**power_seq

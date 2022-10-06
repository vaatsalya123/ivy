# global
from numbers import Number
from typing import Optional, Union, Tuple

import tensorflow as tf
from tensorflow.python.framework.dtypes import DType

import ivy


# Array API Standard #
# ------------------ #


def argmax(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = x.numpy().argmax(axis=axis, keepdims=keepdims)
    return tf.convert_to_tensor(ret, dtype=ret.dtype)


def argmin(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    dtype: DType = tf.dtypes.int64,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    ret = x.numpy().argmin(axis=axis, keepdims=keepdims)
    if dtype is not None:
        return tf.convert_to_tensor(ret, dtype=dtype)
    return tf.convert_to_tensor(ret, dtype=ret.dtype)


def nonzero(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    as_tuple: bool = True,
    size: Optional[int] = None,
    fill_value: Number = 0,
) -> Union[tf.Tensor, tf.Variable, Tuple[Union[tf.Tensor, tf.Variable]]]:
    res = tf.experimental.numpy.nonzero(x)

    if size is not None:
        dtype = tf.int64
        if isinstance(fill_value, float):
            dtype = tf.float64
        res = tf.cast(res, dtype)

        diff = size - res[0].shape[0]
        if diff > 0:
            res = tf.pad(res, [[0, 0], [0, diff]], constant_values=fill_value)
        elif diff < 0:
            res = tf.slice(res, [0, 0], [-1, size])

    if as_tuple:
        return tuple(res)
    return tf.stack(res, axis=1)


def where(
    condition: Union[tf.Tensor, tf.Variable],
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return tf.cast(tf.experimental.numpy.where(condition, x1, x2), x1.dtype)


# Extra #
# ----- #


def argwhere(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    where_x = tf.experimental.numpy.where(x)
    if len(where_x) == 1:
        return tf.expand_dims(where_x[0], -1)
    res = tf.experimental.numpy.concatenate(
        [tf.expand_dims(item, -1) for item in where_x], -1
    )
    return res

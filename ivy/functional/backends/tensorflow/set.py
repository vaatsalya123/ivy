# global
import tensorflow as tf
from typing import Tuple, Union
from collections import namedtuple


def unique_all(
    x: Union[tf.Tensor, tf.Variable]
) -> Tuple[
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
    Union[tf.Tensor, tf.Variable],
]:
    UniqueAll = namedtuple(
        typename="unique_all",
        field_names=["values", "indices", "inverse_indices", "counts"],
    )

    flat_tensor = tf.reshape(x, [-1])
    values, inverse_indices, counts = tf.unique_with_counts(flat_tensor)
    tensor_list = flat_tensor.numpy().tolist()
    if (
        x.dtype.is_floating
        and tf.math.reduce_sum(tf.cast(tf.math.is_nan(values), "float32")).numpy()
    ):
        unique_nan = tf.math.is_nan(values).numpy()

        nan_index = tf.where(tf.math.is_nan(flat_tensor)).numpy().reshape([-1])
        non_nan_index = tf.experimental.numpy.array(
            [tensor_list.index(val) for val in values if not tf.math.is_nan(val)]
        )

        indices = tf.experimental.numpy.full(
            fill_value=float("NaN"), shape=values.shape
        ).numpy()

        indices[unique_nan] = nan_index
        indices[~unique_nan] = non_nan_index
        indices = tf.convert_to_tensor(indices)
    else:
        indices = tf.experimental.numpy.array(
            [tensor_list.index(val) for val in values]
        )
        indices = tf.convert_to_tensor(indices)

    return UniqueAll(
        tf.cast(values, x.dtype),
        tf.cast(indices, dtype="int32"),
        tf.reshape(inverse_indices, x.shape),
        counts,
    )


def unique_counts(
    x: Union[tf.Tensor, tf.Variable],
) -> Tuple[Union[tf.Tensor, tf.Variable], Union[tf.Tensor, tf.Variable]]:
    uc = namedtuple("uc", ["values", "counts"])
    v, _, c = tf.unique_with_counts(tf.reshape(x, [-1]))
    return uc(v, c)


def unique_inverse(
    x: Union[tf.Tensor, tf.Variable],
) -> Tuple[Union[tf.Tensor, tf.Variable], Union[tf.Tensor, tf.Variable]]:
    out = namedtuple("unique_inverse", ["values", "inverse_indices"])
    values, inverse_indices = tf.unique(tf.reshape(x, -1))
    inverse_indices = tf.reshape(inverse_indices, x.shape)
    return out(values, inverse_indices)


def unique_values(x: Union[tf.Tensor, tf.Variable]) -> Union[tf.Tensor, tf.Variable]:
    ret = tf.unique(tf.reshape(x, [-1]))[0]
    return ret

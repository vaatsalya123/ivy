"""Collection of TensorFlow activation functions, wrapped to fit Ivy syntax and
signature.
"""

from typing import Optional, Union

# global
import tensorflow as tf
from tensorflow.python.types.core import Tensor

# local
import ivy


def gelu(
    x: Tensor, /, *, approximate: bool = True, out: Optional[Tensor] = None
) -> Tensor:
    return tf.nn.gelu(x, approximate)


def leaky_relu(
    x: Tensor, /, *, alpha: float = 0.2, out: Optional[Tensor] = None
) -> Tensor:
    return tf.nn.leaky_relu(x, alpha)


def relu(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    return tf.nn.relu(x)


def sigmoid(x: Tensor, /, *, out: Optional[Tensor] = None) -> Tensor:
    if not ivy.is_array(x):
        x = float(x)
    return tf.nn.sigmoid(x)


def softmax(
    x: Tensor, 
    /, 
    *, 
    axis: Optional[int] = None, 
    out: Optional[Tensor] = None
) -> Tensor:
    return tf.nn.softmax(x, axis)


def softplus(
    x: Tensor,
    /,
    *,
    beta: Optional[Union[int, float]] = None,
    threshold: Optional[Union[int, float]] = None,
    out: Optional[Tensor] = None,
) -> Tensor:

    if beta is not None and beta != 1:
        x_beta = x * beta
        res = (tf.nn.softplus(x_beta)) / beta
    else:
        x_beta = x
        res = tf.nn.softplus(x)
    if threshold is not None:
        return tf.where(x_beta > threshold, x, res)
    return res


def log_softmax(
    x: Tensor, /, *, axis: Optional[int] = None, out: Optional[Tensor] = None
):
    return tf.nn.log_softmax(x, axis)

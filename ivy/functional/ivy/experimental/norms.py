from typing import Union, Optional

# local
import ivy
from ivy.utils.backend import current_backend
from ivy.func_wrapper import (
    to_native_arrays_and_back,
    handle_out_argument,
    handle_nestable,
    handle_array_like_without_promotion,
)
from ivy.utils.exceptions import handle_exceptions


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def l2_normalize(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    axis: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Normalizes the input array along the given axis to have L2 norm equal to 1.

    Parameters
    ----------
    x
        Input array.
    axis
        Axis along which to normalize. If ``None``, the whole array is normalized.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The normalized array.

    Examples
    --------
    >>> x = ivy.array([[1., 2.], [3., 4.]])
    >>> ivy.l2_normalize(x, axis=1)
    ivy.array([[0.4472, 0.8944],
               [0.6, 0.8]])
    """
    return current_backend(x).l2_normalize(x, axis=axis, out=out)


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def batch_norm(
    x: Union[ivy.NativeArray, ivy.Array],
    mean: Union[ivy.NativeArray, ivy.Array],
    variance: Union[ivy.NativeArray, ivy.Array],
    /,
    *,
    offset: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    scale: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    training: bool = False,
    eps: float = 1e-5,
    momentum: float = 1e-1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """
    Applies batch normalization to the input array and returns the normalized input,
    running mean and running variance arrays as output. If ``training == False``,
    the mean and variance arrays passed as input are used for normalization
    and the same arrays are returned as running mean and running variance
    respectively. However, when ``training ==True``, this function computes the
    batch mean and batch variance which is then used for normalization.In this case,
    the function returns the running mean and running variance calculated
    using the following formula:

    running_mean = (1 - momentum) * running_mean + momentum * batch_mean
    running_var = (1 - momentum) * running_var + momentum * frac{n}{n-1} * batch_var

    Parameters
    ----------
    x
        Input array of shape (N, *S, C), where N is the batch dimension,
        *S corresponds to any number of spatial dimensions and
         C corresponds to the channel dimension.
    mean
        Mean array used for input's normalization. If ``training=True``
        then it must be one dimensional with size equal to the size of
        channel dimension C. If ``training=False`` then it can be of any
        shape broadcastble to the input shape.
    variance
        Variance array for the input's normalization. If ``training=True``
        then it must be one dimensional with size equal to the size of
        channel dimension C. If ``training=False`` then it can be of any shape
        broadcastble to the input shape.
    offset
        An offset array. If present, will be added to the normalized input.
        If ``training=True`` then it must be one dimensional with size equal
        to the size of channel dimension C. If ``training=False`` then it can
        be of any shape broadcastble to the input shape.
    scale
        A scale array. If present, the scale is applied to the normalized input.
        If ``training=True`` then it must be one dimensional with size equal to
        the size of channel dimension C. If ``training=False`` then it can be of
        any shape broadcastble to the input shape.
    training
        If true, calculate and use the mean and variance of `x`. Otherwise, use the
        provided `mean` and `variance`.
    eps
        A small float number to avoid dividing by 0.
    momentum
         the value used for the running_mean and running_var computation.
          Default value is 0.1.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
         Tuple of arrays containing
          the normalized input, running_mean, and running_variance.
    """
    runningmean = mean
    runningvariance = variance
    if training:
        ndims = len(x.shape)
        numel = x.size if ivy.current_backend_str() != "torch" else x.numel()
        n = numel if ndims == 1 else numel / x.shape[-1]
        dims = (0, *range(1, ndims - 1))
        mean = ivy.mean(x, axis=dims)
        variance = ivy.var(x, axis=dims)
        runningmean = (1 - momentum) * runningmean + momentum * mean
        runningvariance = (1 - momentum) * runningvariance + momentum * variance * n / (
            n - 1
        )
    inv = 1.0 / ivy.sqrt(variance + eps)
    if scale is not None:
        inv = inv * scale
    xnormalized = x * inv.astype(x.dtype, copy=False) + ivy.astype(
        offset - mean * inv if offset is not None else -mean * inv, x.dtype
    )
    return xnormalized, runningmean, runningvariance


batch_norm.mixed_function = True


@handle_nestable
@handle_out_argument
@to_native_arrays_and_back
@handle_exceptions
@handle_array_like_without_promotion
def instance_norm(
    x: Union[ivy.NativeArray, ivy.Array],
    mean: Union[ivy.NativeArray, ivy.Array],
    variance: Union[ivy.NativeArray, ivy.Array],
    /,
    *,
    offset: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    scale: Optional[Union[ivy.NativeArray, ivy.Array]] = None,
    training: bool = False,
    eps: float = 0e-5,
    momentum: float = 1e-1,
    out: Optional[ivy.Array] = None,
):
    """
    Applies instance normalization to the input array and returns the normalized input,
    running mean and running variance arrays as output. If ``training == False``,
    the mean and variance arrays passed as input are used for normalization
    and the same arrays are returned as running mean and running variance
    respectively. However, when ``training ==True``, this function computes the
    mean and variance across the spatial dimensions which is then used for
    normalization.In this case, the function returns the running mean and
    running variance calculated using the following formula:

    running_mean = (1 - momentum) * running_mean + momentum * batch_mean
    running_var = (1 - momentum) * running_var + momentum * frac{n}{n-1} * batch_var

    Parameters
    ----------
    x
        Input array of shape (N, *S, C), where N is the batch dimension,
        *S corresponds to any number of spatial dimensions and
         C corresponds to the channel dimension.
    mean
        Mean array used for input's normalization. If ``training=True``
        then it must be one dimensional with size equal to the size of
        channel dimension C. If ``training=False`` then it can be of any
        shape broadcastble to the input shape.
    variance
        Variance array for the input's normalization. If ``training=True``
        then it must be one dimensional with size equal to the size of
        channel dimension C. If ``training=False`` then it can be of any shape
        broadcastble to the input shape.
    offset
        An offset array. If present, will be added to the normalized input.
        If ``training=True`` then it must be one dimensional with size equal
        to the size of channel dimension C. If ``training=False`` then it can
        be of any shape broadcastble to the input shape.
    scale
        A scale array. If present, the scale is applied to the normalized input.
        If ``training=True`` then it must be one dimensional with size equal to
        the size of channel dimension C. If ``training=False`` then it can be of
        any shape broadcastble to the input shape.
    training
        If true, calculate and use the mean and variance of `x`. Otherwise, use the
        provided `mean` and `variance`.
    eps
        A small float number to avoid dividing by 0.
    momentum
         the value used for the running_mean and running_var computation.
          Default value is 0.1.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
         Tuple of arrays containing
          the normalized input, running_mean, and running_variance.
    """
    N = x.shape[0]
    C = x.shape[-1]
    S = x.shape[1:-1]
    xdims = len(x.shape)
    x = ivy.permute_dims(x, axes=(*range(1, xdims - 1), 0, xdims - 1))
    x = x.reshape((1, *S, N * C))
    mean = ivy.tile(mean, N)
    variance = ivy.tile(variance, N)
    scale = ivy.tile(scale, N)
    offset = ivy.tile(offset, N)
    xnormalized, runningmean, runningvariance = batch_norm(
        x,
        mean,
        variance,
        scale=scale,
        offset=offset,
        training=training,
        eps=eps,
        momentum=momentum,
        out=out,
    )
    xnormalized = xnormalized.reshape((*S, N, C))
    return (
        ivy.permute_dims(
            xnormalized, axes=(xdims - 2, *range(0, xdims - 2), xdims - 1)
        ),
        runningmean.reshape((N, C)).mean(axis=0),
        runningvariance.reshape((N, C)).mean(axis=0),
    )


instance_norm.mixed_function = True


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def lp_normalize(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    p: float = 2,
    axis: Optional[int] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Normalizes the input array along the given axis to have Lp norm equal to 1.

    Parameters
    ----------
    x
        Input array.
    p
        The Lp norm to use for normalization. Default is L2 norm (p=2).
    axis
        Axis along which to normalize. If ``None``, the whole array is normalized.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        The normalized array.

    Examples
    --------
    >>> x = ivy.array([[1., 2.], [3., 4.]])
    >>> ivy.lp_normalize(x, p=1, axis=1)
    ivy.array([[0.3333, 0.6666],
               [0.75, 1.]])
    """
    return current_backend(x).lp_normalize(x, p=p, axis=axis, out=out)

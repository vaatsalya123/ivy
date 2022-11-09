from typing import (
    Optional,
    Union,
    Tuple,
)
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
)
from ivy.exceptions import handle_exceptions


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def median(
    input: ivy.Array,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Compute the median along the specified axis.

    Parameters
    ----------
    input
        Input array.
    axis
        Axis or axes along which the medians are computed. The default is to compute
        the median along a flattened version of the array.
    keepdims
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The median of the array elements.

    Functional Examples
    -------------------
    >>> a = ivy.array([[10, 7, 4], [3, 2, 1]])
    >>> ivy.median(a)
    3.5
    >>> ivy.median(a, axis=0)
    ivy.array([6.5, 4.5, 2.5])
    """
    return ivy.current_backend().median(input, axis=axis, keepdims=keepdims, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def nanmean(
    a: ivy.Array,
    /,
    *,
    axis: Optional[Union[Tuple[int], int]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[Union[ivy.Dtype, ivy.NativeDtype]] = None,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the mean of all non-NaN elements along the specified dimensions.

    Parameters
    ----------
    a
        Input array.
    axis
        Axis or axes along which the means are computed.
        The default is to compute the mean of the flattened array.
    keepdims
        If this is set to True, the axes which are reduced are left in the result
        as dimensions with size one. With this option, the result will broadcast
        correctly against the original a. If the value is anything but the default,
        then keepdims will be passed through to the mean or sum methods of sub-classes
        of ndarray. If the sub-classes methods does not implement keepdims any
        exceptions will be raised.
    dtype
        The desired data type of returned tensor. Default is None.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        The nanmean of the array elements.

    Functional Examples
    -------------------
    >>> a = ivy.array([[1, ivy.nan], [3, 4]])
    >>> ivy.nanmean(a)
    2.6666666666666665
    >>> ivy.nanmean(a, axis=0)
    ivy.array([2.,  4.])
    """
    return ivy.current_backend(a).nanmean(
        a, axis=axis, keepdims=keepdims, dtype=dtype, out=out
    )


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def unravel_index(
    indices: ivy.Array,
    shape: Tuple[int],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Converts a flat index or array of flat indices
    into a tuple of coordinate arrays.

    Parameters
    ----------
    indices
        Input array.
    shape
        The shape of the array to use for unraveling indices.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Tuple with arrays that have the same shape as the indices array.

    Functional Examples
    -------------------
    >>> indices = ivy.array([22, 41, 37])
    >>> ivy.unravel_index(indices, (7,6))
    (ivy.array([3, 6, 6]), ivy.array([4, 5, 1]))
    """
    return ivy.current_backend(indices).unravel_index(indices, shape, out=out)

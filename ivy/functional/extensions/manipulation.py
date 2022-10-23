from typing import (
    Optional,
    Union,
    Tuple,
    Iterable,
    Sequence,
    Generator,
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
def flatten(
    x: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    start_dim: Optional[int] = 0,
    end_dim: Optional[int] = -1,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Flattens input by reshaping it into a one-dimensional tensor.
        If start_dim or end_dim are passed, only dimensions starting
        with start_dim and ending with end_dim are flattened.
        The order of elements in input is unchanged.

    Parameters
    ----------
    x
        input array to flatten.
    start_dim
        first dim to flatten. If not set, defaults to 0.
    end_dim
        last dim to flatten. If not set, defaults to -1.

    Returns
    -------
    ret
        the flattened array over the specified dimensions.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.manipulation_functions.concat.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :class:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x = np.array([1,2], [3,4])
    >>> ivy.flatten(x)
    ivy.array([1, 2, 3, 4])

    >>> x = np.array(
        [[[[ 5,  5,  0,  6],
         [17, 15, 11, 16],
         [ 6,  3, 13, 12]],

        [[ 6, 18, 10,  4],
         [ 5,  1, 17,  3],
         [14, 14, 18,  6]]],


       [[[12,  0,  1, 13],
         [ 8,  7,  0,  3],
         [19, 12,  6, 17]],

        [[ 4, 15,  6, 15],
         [ 0,  5, 17,  9],
         [ 9,  3,  6, 19]]],


       [[[17, 13, 11, 16],
         [ 4, 18, 17,  4],
         [10, 10,  9,  1]],

        [[19, 17, 13, 10],
         [ 4, 19, 16, 17],
         [ 2, 12,  8, 14]]]]
         )
    >>> ivy.flatten(x, start_dim = 1, end_dim = 2)
    ivy.array(
        [[[ 5,  5,  0,  6],
          [17, 15, 11, 16],
          [ 6,  3, 13, 12],
          [ 6, 18, 10,  4],
          [ 5,  1, 17,  3],
          [14, 14, 18,  6]],

         [[12,  0,  1, 13],
          [ 8,  7,  0,  3],
          [19, 12,  6, 17],
          [ 4, 15,  6, 15],
          [ 0,  5, 17,  9],
          [ 9,  3,  6, 19]],

         [[17, 13, 11, 16],
          [ 4, 18, 17,  4],
          [10, 10,  9,  1],
          [19, 17, 13, 10],
          [ 4, 19, 16, 17],
          [ 2, 12,  8, 14]]]))
    """
    if start_dim == end_dim and len(x.shape) != 0:
        return x
    if start_dim not in range(-len(x.shape), len(x.shape)):
        raise IndexError(
            f"Dimension out of range (expected to be in range of\
            {[-len(x.shape), len(x.shape) - 1]}, but got {start_dim}"
        )
    if end_dim not in range(-len(x.shape), len(x.shape)):
        raise IndexError(
            f"Dimension out of range (expected to be in range of\
            {[-len(x.shape), len(x.shape) - 1]}, but got {end_dim}"
        )
    if start_dim < 0:
        start_dim = len(x.shape) + start_dim
    if end_dim < 0:
        end_dim = len(x.shape) + end_dim
    c = 1
    for i in range(start_dim, end_dim + 1):
        c *= x.shape[i]
    lst = [c]
    if start_dim != 0:
        for i in range(0, start_dim):
            lst.insert(i, x.shape[i])
    for i in range(end_dim + 1, len(x.shape)):
        lst.insert(i, x.shape[i])
    return ivy.reshape(x, tuple(lst))


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def moveaxis(
    a: Union[ivy.Array, ivy.NativeArray],
    source: Union[int, Sequence[int]],
    destination: Union[int, Sequence[int]],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Move axes of an array to new positions..

    Parameters
    ----------
    a
        The array whose axes should be reordered.
    source
        Original positions of the axes to move. These must be unique.
    destination
        Destination positions for each of the original axes.
        These must also be unique.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array with moved axes. This array is a view of the input array.

    Examples
    --------
    With :class:`ivy.Array` input:
    >>> x = ivy.zeros((3, 4, 5))
    >>> ivy.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> ivy.moveaxis(x, -1, 0).shape
    (5, 3, 4)
    """
    return ivy.current_backend().moveaxis(a, source, destination, out=out)


@handle_exceptions
def ndenumerate(
    input: Iterable,
) -> Generator:
    """Multidimensional index iterator.

    Parameters
    ----------
    input
        Input array to iterate over.

    Returns
    -------
    ret
        An iterator yielding pairs of array coordinates and values.

    Examples
    --------
    >>> a = ivy.array([[1, 2], [3, 4]])
    >>> for index, x in ivy.ndenumerate(a):
    >>>     print(index, x)
    (0, 0) 1
    (0, 1) 2
    (1, 0) 3
    (1, 1) 4
    """

    def _ndenumerate(input, t=None):
        if t is None:
            t = ()
        if not hasattr(input, "__iter__"):
            yield t, input
        else:
            for i, v in enumerate(input):
                yield from _ndenumerate(v, t + (i,))

    return _ndenumerate(input)


@handle_exceptions
def ndindex(
    shape: Tuple,
) -> Generator:
    """Multidimensional index iterator.

    Parameters
    ----------
    shape
        The shape of the array to iterate over.

    Returns
    -------
    ret
        An iterator yielding array coordinates.

    Examples
    --------
    >>> a = ivy.array([[1, 2], [3, 4]])
    >>> for index in ivy.ndindex(a):
    >>>     print(index)
    (0, 0)
    (0, 1)
    (1, 0)
    (1, 1)
    """

    def _iter_product(*args, repeat=1):
        pools = [tuple(pool) for pool in args] * repeat
        result = [[]]
        for pool in pools:
            result = [x + [y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)

    args = []
    for s in range(len(shape)):
        args += [range(shape[s])]
    return _iter_product(*args)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def heaviside(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Computes the Heaviside step function for each element in x1.

    Parameters
    ----------
    x1
        input array.
    x2
        values to use where x1 is zero.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        output array with element-wise Heaviside step function of x1.
        This is a scalar if both x1 and x2 are scalars.

    Examples
    --------
    With :class:`ivy.Array` input:

    >>> x1 = ivy.array([-1.5, 0, 2.0])
    >>> x2 = ivy.array([0.5])
    >>> ivy.heaviside(x1, x2)
    ivy.array([0.0000, 0.5000, 1.0000])

    >>> x1 = ivy.array([-1.5, 0, 2.0])
    >>> x2 = ivy.array([1.2, -2.0, 3.5])
    >>> ivy.heaviside(x1, x2)
    ivy.array([0., -2., 1.])
    """
    return ivy.current_backend().heaviside(x1, x2, out=out)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def flipud(
    m: Union[ivy.Array, ivy.NativeArray],
    /,
    *,
    out: Optional[Union[ivy.Array, ivy.NativeArray]] = None,
) -> Union[ivy.Array, ivy.NativeArray]:
    """Flip array in the up/down direction.
    Flip the entries in each column in the up/down direction.
    Rows are preserved, but appear in a different order than before.

    Parameters
    ----------
    m
        The array to be flipped.
    out
        optional output array, for writing the result to.

    Returns
    -------
    ret
        Array corresponding to input array with elements
        order reversed along axis 0.

    Examples
    --------
    >>> m = ivy.diag([1, 2, 3])
    >>> ivy.flipud(m)
    ivy.array([[ 0.,  0.,  3.],
        [ 0.,  2.,  0.],
        [ 1.,  0.,  0.]])
    """
    return ivy.current_backend().flipud(m, out=out)


@to_native_arrays_and_back
@handle_out_argument
def vstack(arrays: Sequence[ivy.Array], /) -> ivy.Array:
    """Stack arrays in sequence vertically (row wise).
    Parameters
    ----------
    arrays
        Sequence of arrays to be stacked.
    Returns
    -------
    ret
        The array formed by stacking the given arrays.
    Examples
    --------
    >>> x = ivy.array([1, 2, 3])
    >>> y = ivy.array([2, 3, 4])
    >>> ivy.vstack((x, y))
    ivy.array([[ 1.,  2.,  3.],
        [ 2.,  3.,  4.]])
    >>> ivy.vstack((x, y, x, y))
    ivy.array([[ 1.,  2.,  3.],
        [ 2.,  3.,  4.],
        [ 1.,  2.,  3.],
        [ 2.,  3.,  4.]])
    """
    return ivy.current_backend(arrays[0]).vstack(arrays)

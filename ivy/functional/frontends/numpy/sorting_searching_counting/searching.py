# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@to_ivy_arrays_and_back
def where(cond, x1=None, x2=None, /):
    if x1 is None and x2 is None:
        # numpy where behaves as np.asarray(condition).nonzero() when x and y
        # not included
        return ivy.asarray(cond).nonzero()
    elif x1 is not None and x2 is not None:
        return ivy.where(cond, x1, x2)
    else:
        raise ivy.exceptions.IvyException("where takes either 1 or 3 arguments")


@to_ivy_arrays_and_back
def nonzero(a):
    return ivy.nonzero(a)


@to_ivy_arrays_and_back
def argmin(a, /, *, axis=None, keepdims=False, out=None):
    return ivy.argmin(a, axis=axis, out=out, keepdims=keepdims)


@to_ivy_arrays_and_back
def argmax(
    a,
    /,
    *,
    axis=None,
    out=None,
    keepdims=False,
):
    return ivy.argmax(a, axis=axis, out=out, keepdims=keepdims)


@to_ivy_arrays_and_back
def flatnonzero(a):
    return ivy.nonzero(ivy.reshape(a, (-1,)))


@to_ivy_arrays_and_back
def searchsorted(a, v, side="left", sorter=None):
    return ivy.searchsorted(a, v, side=side, sorter=sorter)

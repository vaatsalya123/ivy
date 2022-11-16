# global
import ivy
from ivy.func_wrapper import from_zero_dim_arrays_to_float
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back


@from_zero_dim_arrays_to_float
@to_ivy_arrays_and_back
def mean(
        x,
        /,
        *,
        axis=None,
        keepdims=False,
        out=None,
        dtype=None,
        where=True,
):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))

    ret = ivy.mean(x, axis=axis, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    return ret


@from_zero_dim_arrays_to_float
@to_ivy_arrays_and_back
def nanmean(
        a,
        /,
        *,
        axis=None,
        keepdims=False,
        out=None,
        dtype=None,
        where=True,
):
    is_nan = ivy.isnan(a)
    axis = tuple(axis) if isinstance(axis, list) else axis

    if not ivy.any(is_nan):
        if dtype:
            a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))
        ret = ivy.mean(a, axis=axis, keepdims=keepdims, out=out)

        if ivy.is_array(where):
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    else:
        a = [i for i in a if ivy.isnan(i) is False]

        if dtype:
            a = ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype))
        ret = ivy.mean(a, axis=axis, keepdims=keepdims, out=out)

        if ivy.is_array(where):
            ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    return ret


@from_zero_dim_arrays_to_float
def std(
        x,
        /,
        *,
        axis=None,
        correction=0.0,
        keepdims=False,
        out=None,
        dtype=None,
        where=True,
):
    axis = tuple(axis) if isinstance(axis, list) else axis
    if dtype:
        x = ivy.astype(ivy.array(x), ivy.as_ivy_dtype(dtype))

    ret = ivy.std(x, axis=axis, correction=correction, keepdims=keepdims, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)

    return ret

# global
import math
import ivy

# local
from ivy.func_wrapper import from_zero_dim_arrays_to_float
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
)


def convolve(a, v, mode="full"):
    pass


@from_zero_dim_arrays_to_float
@handle_numpy_casting
@to_ivy_arrays_and_back
def clip(
    a,
    a_min,
    a_max,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="k",
    dtype=None,
    subok=True,
):
    ivy.assertions.check_all_or_any_fn(
        a_min,
        a_max,
        fn=ivy.exists,
        type="any",
        limit=[1, 2],
        message="at most one of a_min and a_max can be None",
    )
    a = ivy.array(a, dtype=dtype)
    if a_min is None:
        ret = ivy.minimum(a, a_max, out=out)
    elif a_max is None:
        ret = ivy.maximum(a, a_min, out=out)
    else:
        ret = ivy.clip(a, a_min, a_max, out=out)
    if ivy.is_array(where):
        ret = ivy.where(where, ret, ivy.default(out, ivy.zeros_like(ret)), out=out)
    return ret


@from_zero_dim_arrays_to_float
@handle_numpy_casting
@to_ivy_arrays_and_back
def sqrt(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    x = ivy.array(x)
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), ivy.sqrt(x), ivy.default(out, x), out=out
    )
    return ret


@from_zero_dim_arrays_to_float
@handle_numpy_casting
@to_ivy_arrays_and_back
def cbrt(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    all_positive = ivy.pow(ivy.abs(x), 1.0 / 3.0)
    fixed_signs = ivy.where(ivy.less(x, 0.0), ivy.negative(all_positive), all_positive)
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), fixed_signs, ivy.default(out, x), out=out
    )
    return ret


@from_zero_dim_arrays_to_float
@handle_numpy_casting
@to_ivy_arrays_and_back
def square(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), ivy.square(x), ivy.default(out, x), out=out
    )
    return ret


@from_zero_dim_arrays_to_float
@handle_numpy_casting
@to_ivy_arrays_and_back
def absolute(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), ivy.abs(x), ivy.default(out, x), out=out
    )
    return ret


@from_zero_dim_arrays_to_float
@handle_numpy_casting
@to_ivy_arrays_and_back
def fabs(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    ret = ivy.where(
        ivy.broadcast_to(where, x.shape), ivy.abs(x), ivy.default(out, x), out=out
    )
    return ret


@from_zero_dim_arrays_to_float
@handle_numpy_casting
@to_ivy_arrays_and_back
def sign(
    x,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    ret = ivy.sign(x, out=out)
    if where is not None:
        ret = ivy.where(
            ivy.broadcast_to(where, x.shape), ret, ivy.default(out, x), out=out
        )
    return ret


@from_zero_dim_arrays_to_float
@handle_numpy_casting
@to_ivy_arrays_and_back
def heaviside(
    x1,
    x2,
    /,
    out=None,
    *,
    where=True,
    casting="same_kind",
    order="K",
    dtype=None,
    subok=True,
):
    x1 = ivy.array(x1)
    x2 = ivy.array(x2)
    ret = ivy.where(
        ivy.equal(x1, x1.full_like(0.0)),
        x2,
        ivy.where(ivy.less(x1, 0.0), ivy.zeros_like(x1), ivy.ones_like(x1)),
    )
    ret = ivy.where(
        ivy.broadcast_to(where, x1.shape), ret, ivy.default(out, x1), out=out
    )
    return ret


@to_ivy_arrays_and_back
def nan_to_num(x, copy=True, nan=0.0, posinf=None, neginf=None):
    ret = ivy.array(x, copy=copy)
    bounds = ivy.finfo(x)
    pinf = posinf if posinf is not None else bounds.max
    ninf = neginf if neginf is not None else bounds.min
    ivy.where(ivy.equal(ret, ret.full_like(math.nan)), ret.full_like(nan), ret, out=ret)
    ivy.where(
        ivy.equal(ret, ret.full_like(math.inf)), ret.full_like(pinf), ret, out=ret
    )
    ivy.where(
        ivy.equal(ret, ret.full_like(-math.inf)), ret.full_like(ninf), ret, out=ret
    )
    return ret


@to_ivy_arrays_and_back
def real_if_close(a, tol=100):
    return ivy.array(a)  # ivy doesn't yet support complex numbers


@to_ivy_arrays_and_back
def interp(x, xp, fp, left=None, right=None, period=None):
    x_arr = ivy.array(x)
    fix_later = False
    if x_arr.shape == ():
        x_arr = ivy.array([x])
        fix_later = True
    x = ivy.astype(x_arr, "float64")
    xp = ivy.astype(ivy.array(xp), "float64")
    fp = ivy.astype(ivy.array(fp), "float64")
    ivy.assertions.check_equal(xp.ndim, 1)
    ivy.assertions.check_equal(fp.ndim, 1)
    ivy.assertions.check_equal(xp.shape[0], fp.shape[0])
    if period is not None:
        ivy.assertions.check_equal(period, 0, inverse=True)
        period = ivy.abs(period)
        x = ivy.remainder(x, period)
        xp = ivy.remainder(xp, period)
        asort_xp = ivy.argsort(xp)
        xp = xp[asort_xp]
        fp = fp[asort_xp]
        xp = ivy.concat((xp[-1:] - period, xp, xp[0:1] + period))
        fp = ivy.concat((fp[-1:], fp, fp[0:1]))

    def interp_inner(value):
        if value < xp[0]:
            return left if left is not None else fp[0]
        elif value > xp[-1]:
            return right if right is not None else fp[-1]
        else:
            last = None
            if xp.shape[0] < 3:
                for i in range(xp.shape[0]):
                    if xp[i] == value:
                        return fp[i]
                    elif xp[i] < value:
                        last = i
            else:
                first = 0
                last = xp.shape[0]
                while first <= last:
                    midpoint = (first + last) // 2
                    if xp[midpoint] == value:
                        return fp[midpoint]
                    else:
                        if value < xp[midpoint]:
                            last = midpoint - 1
                        else:
                            first = midpoint + 1
            dist = (value - xp[last]) / (xp[last + 1] - xp[last])
            return (fp[last + 1] - fp[last]) * dist + fp[last]

    ret = ivy.map(interp_inner, unique={"value": x})
    if fix_later:
        return ivy.astype(ivy.array(ret[0]), "float64")
    else:
        return ivy.astype(ivy.array(ret), "float64")

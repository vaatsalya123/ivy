from typing import Optional, Union, Tuple
from ivy.functional.backends.jax import JaxArray
import jax.numpy as jnp


def lcm(x1: JaxArray, x2: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.lcm(x1, x2)


def sinc(x: JaxArray, /, *, out: Optional[JaxArray] = None) -> JaxArray:
    return jnp.sinc(x)


def fmod(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.fmod(x1, x2)


def fmax(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.fmax(x1, x2)


def trapz(
    y: JaxArray,
    /,
    *,
    x: Optional[JaxArray] = None,
    dx: Optional[float] = 1.0,
    axis: Optional[int] = -1,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.trapz(y, x=x, dx=dx, axis=axis)


def float_power(
    x1: Union[JaxArray, float, list, tuple],
    x2: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.float_power(x1, x2)


def exp2(
    x: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.exp2(x)


def count_nonzero(
    a: JaxArray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[jnp.dtype] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    if dtype is None:
        return jnp.count_nonzero(a, axis=axis, keepdims=keepdims)
    return jnp.array(jnp.count_nonzero(a, axis=axis, keepdims=keepdims), dtype=dtype)


def nansum(
    x: JaxArray,
    /,
    *,
    axis: Optional[Union[tuple, int]] = None,
    dtype: Optional[jnp.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.nansum(x, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


def gcd(
    x1: Union[JaxArray, float, list, tuple],
    x2: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.gcd(x1, x2)


def isclose(
    a: JaxArray,
    b: JaxArray,
    /,
    *,
    rtol: Optional[float] = 1e-05,
    atol: Optional[float] = 1e-08,
    equal_nan: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def isposinf(
    x: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.isposinf(x, out=out)


def isneginf(
    x: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.isneginf(x, out=out)


def nan_to_num(
    x: JaxArray,
    /,
    *,
    copy: Optional[bool] = True,
    nan: Optional[Union[float, int]] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)


def logaddexp2(
    x1: Union[JaxArray, float, list, tuple],
    x2: Union[JaxArray, float, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.logaddexp2(x1, x2)


def signbit(
    x: Union[JaxArray, float, int, list, tuple],
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.signbit(x)


def allclose(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    rtol: Optional[float] = 1e-05,
    atol: Optional[float] = 1e-08,
    equal_nan: Optional[bool] = False,
    out: Optional[JaxArray] = None,
) -> bool:
    return jnp.allclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


def diff(
    x: Union[JaxArray, int, float, list, tuple], /, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jnp.diff(x, out=out)


def fix(
    x: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.fix(x, out=out)


def nextafter(
    x1: JaxArray,
    x2: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.nextafter(x1, x2)


def zeta(
    x: JaxArray,
    q: JaxArray,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    inf_indices = jnp.union1d(
        jnp.array(jnp.where(x == 1.0)), jnp.array(jnp.where(q <= 0))
    )
    nan_indices = jnp.where(x <= 0)
    n, res = 1, 1 / q**x
    while n < 10000:
        term = 1 / (q + n) ** x
        n, res = n + 1, res + term
    ret = jnp.round(res, decimals=4)
    ret = ret.at[inf_indices].set(jnp.inf)
    ret = ret.at[nan_indices].set(jnp.nan)
    return ret

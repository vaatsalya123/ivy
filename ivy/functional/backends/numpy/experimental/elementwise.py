from typing import Optional, Union, Tuple
import numpy as np
from ivy.functional.backends.numpy.helpers import _scalar_output_to_0d_array
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version


@_scalar_output_to_0d_array
@with_unsupported_dtypes({"1.23.0 and below": ("bfloat16",)}, backend_version)
def sinc(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    return np.sinc(x).astype(x.dtype)


@_scalar_output_to_0d_array
def lcm(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.abs(
        np.lcm(
            x1,
            x2,
            out=out,
        )
    )


lcm.support_native_out = True


@_scalar_output_to_0d_array
def fmod(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.fmod(
        x1,
        x2,
        out=None,
    )


fmod.support_native_out = True


@_scalar_output_to_0d_array
def fmax(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.fmax(
        x1,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    )


fmax.support_native_out = True


@_scalar_output_to_0d_array
def trapz(
    y: np.ndarray,
    /,
    *,
    x: Optional[np.ndarray] = None,
    dx: Optional[float] = 1.0,
    axis: Optional[int] = -1,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.trapz(y, x=x, dx=dx, axis=axis)


trapz.support_native_out = False


def float_power(
    x1: Union[np.ndarray, float, list, tuple],
    x2: Union[np.ndarray, float, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.asarray(np.float_power(x1, x2, out=out), dtype=x1.dtype)


float_power.support_native_out = True


def exp2(
    x: Union[np.ndarray, float, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.exp2(x, out=out)


exp2.support_native_out = True


@_scalar_output_to_0d_array
def count_nonzero(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[np.dtype] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    if isinstance(axis, list):
        axis = tuple(axis)
    if dtype is None:
        return np.count_nonzero(x, axis=axis, keepdims=keepdims)
    return np.array(np.count_nonzero(x, axis=axis, keepdims=keepdims), dtype=dtype)


count_nonzero.support_native_out = False


def nansum(
    x: np.ndarray,
    /,
    *,
    axis: Optional[Union[tuple, int]] = None,
    dtype: Optional[np.dtype] = None,
    keepdims: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.nansum(x, axis=axis, dtype=dtype, keepdims=keepdims, out=out)


nansum.support_native_out = True


def gcd(
    x1: Union[np.ndarray, int, list, tuple],
    x2: Union[np.ndarray, float, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.gcd(x1, x2, out=out)


gcd.support_native_out = True


def isclose(
    a: np.ndarray,
    b: np.ndarray,
    /,
    *,
    rtol: Optional[float] = 1e-05,
    atol: Optional[float] = 1e-08,
    equal_nan: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


isclose.support_native_out = False


def isposinf(
    x: Union[np.ndarray, float, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.isposinf(x, out=out)


isposinf.support_native_out = True


def isneginf(
    x: Union[np.ndarray, float, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.isneginf(x, out=out)


isneginf.support_native_out = True


def nan_to_num(
    x: np.ndarray,
    /,
    *,
    copy: Optional[bool] = True,
    nan: Optional[Union[float, int]] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.nan_to_num(x, copy=copy, nan=nan, posinf=posinf, neginf=neginf)


nan_to_num.support_native_out = False


def logaddexp2(
    x1: Union[np.ndarray, int, list, tuple],
    x2: Union[np.ndarray, int, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.logaddexp2(x1, x2, out=out)


logaddexp2.support_native_out = True


def signbit(
    x: Union[np.ndarray, float, int, list, tuple],
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.signbit(x, out=out)


signbit.support_native_out = True


def allclose(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    rtol: Optional[float] = 1e-05,
    atol: Optional[float] = 1e-08,
    equal_nan: Optional[bool] = False,
    out: Optional[np.ndarray] = None,
) -> bool:
    return np.allclose(x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


isclose.support_native_out = False


def fix(
    x: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.fix(x, out=out)


fix.support_native_out = True


def nextafter(
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.nextafter(x1, x2)


nextafter.support_natvie_out = True

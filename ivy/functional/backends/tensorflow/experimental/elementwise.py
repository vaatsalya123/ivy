import operator
from typing import Union, Optional, Tuple, List
import tensorflow as tf
from .. import backend_version


# local
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
import tensorflow_probability as tfp
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()


def sinc(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    return tf.cast(tf.experimental.numpy.sinc(x), x.dtype)


@with_unsupported_dtypes({"2.9.1 and below": ("unsigned",)}, backend_version)
def lcm(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.abs(tf.experimental.numpy.lcm(x1, x2))


def fmod(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    result = tf.math.floormod(x1, x2, name=None)
    temp = [result, x1]
    return tf.map_fn(
        lambda x: x[0] if (x[0] * x[1] >= 0) else (-1 * x[0]),
        temp,
        fn_output_signature=result.dtype,
    )


def fmax(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    temp = tf.constant(float("nan"))
    tf.dtypes.cast(x1, tf.float64)
    tf.dtypes.cast(x2, tf.float64)
    x1 = tf.where(tf.math.is_nan(x1, temp), x2, x1)
    x2 = tf.where(tf.math.is_nan(x2, temp), x1, x2)
    tf.experimental.numpy.experimental_enable_numpy_behavior()
    ret = tf.experimental.numpy.maximum(x1, x2)
    return ret


def trapz(
    y: Union[tf.Tensor, tf.Variable],
    /,
    *,
    x: Optional[Union[tf.Tensor, tf.Variable]] = None,
    dx: Optional[float] = 1.0,
    axis: Optional[int] = -1,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tfp.math.trapz(y, x=x, dx=dx, axis=axis, name=None)


def float_power(
    x1: Union[tf.Tensor, tf.Variable, float, list, tuple],
    x2: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.float_power(x1, x2)


def exp2(
    x: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.pow(2, x, name=None)


def count_nonzero(
    a: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: Optional[bool] = False,
    dtype: Optional[tf.DType] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    if dtype is None:
        return tf.math.count_nonzero(a, axis=axis, keepdims=keepdims, name=None)
    return tf.math.count_nonzero(
        a, axis=axis, keepdims=keepdims, dtype=dtype, name=None
    )


def nansum(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    axis: Optional[Union[tuple, int]] = None,
    dtype: Optional[tf.DType] = None,
    keepdims: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.nansum(x, axis=axis, dtype=dtype, keepdims=keepdims)


@with_unsupported_dtypes(
    {"2.9.1 and below": ("uint8", "uint16", "uint32", "uint64")}, backend_version
)
def gcd(
    x1: Union[tf.Tensor, tf.Variable, int, list, tuple],
    x2: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.gcd(x1, x2)


def isclose(
    a: Union[tf.Tensor, tf.Variable],
    b: Union[tf.Tensor, tf.Variable],
    /,
    *,
    rtol: Optional[float] = 1e-05,
    atol: Optional[float] = 1e-08,
    equal_nan: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.isclose(
        a, b, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


def isposinf(
    x: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.isposinf(x)


def isneginf(
    x: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.isneginf(x)


def nan_to_num(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    copy: Optional[bool] = True,
    nan: Optional[Union[float, int]] = 0.0,
    posinf: Optional[Union[float, int]] = None,
    neginf: Optional[Union[float, int]] = None,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    posinf = posinf if posinf is not None else 1.79769313e308
    neginf = neginf if neginf is not None else -1.79769313e308
    ret = tf.where(tf.math.is_nan(x), nan, x)
    ret = tf.where(tf.math.logical_and(tf.math.is_inf(ret), ret > 0), posinf, ret)
    ret = tf.where(tf.math.logical_and(tf.math.is_inf(ret), ret < 0), neginf, ret)
    if copy:
        return ret
    else:
        x = ret
        return x


@with_unsupported_dtypes(
    {
        "2.9.1 and below": (
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
        )
    },
    backend_version,
)
def logaddexp2(
    x1: Union[tf.Tensor, tf.Variable, float, list, tuple],
    x2: Union[tf.Tensor, tf.Variable, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    dtype = x1.dtype
    x1 = tf.cast(x1, tf.float64)
    x2 = tf.cast(x2, tf.float64)
    return ivy.log2(ivy.exp2(x1) + ivy.exp2(x2)).astype(dtype)


def signbit(
    x: Union[tf.Tensor, tf.Variable, float, int, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.signbit(x)


def allclose(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    rtol: Optional[float] = 1e-05,
    atol: Optional[float] = 1e-08,
    equal_nan: Optional[bool] = False,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> bool:
    return tf.experimental.numpy.allclose(
        x1, x2, rtol=rtol, atol=atol, equal_nan=equal_nan
    )


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16,")}, backend_version)
def fix(
    x: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.fix(x)


@with_unsupported_dtypes({"2.9.1 and below": ("float16,")}, backend_version)
def nextafter(
    x1: Union[tf.Tensor, tf.Variable],
    x2: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.nextafter(x1, x2)


@with_unsupported_dtypes(
    {"2.9.1 and below": ("uint8", "uint16", "uint32", "uint64")}, backend_version
)
def diff(
    x: Union[tf.Tensor, tf.Variable, int, float, list, tuple],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.experimental.numpy.diff(x)


@with_unsupported_dtypes({"2.9.1 and below": ("bfloat16, float16,")}, backend_version)
def zeta(
    x: Union[tf.Tensor, tf.Variable],
    q: Union[tf.Tensor, tf.Variable],
    /,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    return tf.math.zeta(x, q)


def _normalize_axis_index(ax: int, ndim: int) -> int:
    if ax >= ndim or ax < -ndim:
        raise ValueError("axis index is out of range")
    return (ax + ndim) % ndim


def _normalize_axis_tuple(axis: Union[int, list, tuple], ndim: int) -> Tuple[int, ...]:
    if type(axis) not in (tuple, list):
        try:
            axis = [operator.index(axis)]
        except TypeError:
            pass
    axis = tuple([_normalize_axis_index(ax, ndim) for ax in axis])
    if len(set(axis)) != len(axis):
        raise ValueError("repeated axis")
    return axis


def gradient(
    x: tf.Tensor,
    /,
    *,
    spacing: Optional[Union[int, list, tuple]] = 1,
    axis: Optional[Union[int, list, tuple]] = None,
    edge_order: Optional[int] = 1,
) -> Union[tf.Tensor, List[tf.Tensor]]:
    """https://github.com/numpy/numpy/blob/v1.23.0/numpy/lib/
    function_base.py#L969-L1312"""
    device = x.device
    x = tf.experimental.numpy.asanyarray(x)
    N = x.ndim  # number of dimensions
    if axis is None:
        axes = tuple(range(N))
    else:
        axes = _normalize_axis_tuple(axis, N)

    len_axes = len(axes)
    n = (
        -1
        if spacing is None
        else (0 if type(spacing) in (int, float) else len(spacing))
    )
    if n == -1:
        # no spacing argument - use 1 in all axes
        dx = [1.0] * len_axes
    elif n == 0:
        dx = [spacing] * len_axes
    elif n == 1 and tf.experimental.numpy.ndim(spacing[0]) == 0:
        # single scalar for all axes
        dx = spacing * len_axes
    elif n == len_axes:
        # scalar or 1d array for each axis
        dx = list(spacing)
        for i, distances in enumerate(dx):
            distances = tf.experimental.numpy.asanyarray(distances)
            if distances.ndim == 0:
                continue
            elif distances.ndim != 1:
                raise ValueError("distances must be either scalars or 1d")
            if len(distances) != x.shape[axes[i]]:
                raise ValueError(
                    "when 1d, distances must match "
                    "the length of the corresponding dimension {} {}".format(
                        len(distances), x.shape[axes[i]]
                    )
                )
            if distances.dtype.is_integer:
                # Convert numpy integer types to float64 to avoid modular
                # arithmetic in np.diff(distances).
                distances = distances.astype(tf.experimental.numpy.float64)
            diffx = tf.experimental.numpy.diff(distances)
            # if distances are constant reduce to the scalar case
            # since it brings a consistent speedup
            # cmp = diffx == diffx[0]
            if tf.reduce_all(tf.equal(diffx, diffx[0])):
                diffx = diffx[0]
            # if tf.reduce_sum(tf.cast(cmp, tf.int32)) == cmp.numel():
            #     print(diffx, (diffx == diffx[0]))
            #     diffx = diffx[0]
            dx[i] = diffx
    else:
        raise TypeError("invalid number of arguments")

    if edge_order > 2:
        raise ValueError("'edge_order' greater than 2 not supported")

    # use central differences on interior and one-sided differences on the
    # endpoints. This preserves second order-accuracy over the full domain.

    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice1 = [slice(None)] * N
    slice2 = [slice(None)] * N
    slice3 = [slice(None)] * N
    slice4 = [slice(None)] * N

    if x.dtype.is_integer:
        x = x.astype(tf.experimental.numpy.float64)

    for axis, ax_dx in zip(axes, dx):
        if x.shape[axis] < edge_order + 1:
            raise ValueError(
                "Shape of array too small to calculate a numerical gradient, "
                "at least (edge_order + 1) elements are required."
            )
        # result allocation
        out = x.numpy()

        # spacing for the current axis
        uniform_spacing = tf.experimental.numpy.ndim(ax_dx) == 0

        # Numerical differentiation: 2nd order interior
        slice1[axis] = slice(1, -1)
        slice2[axis] = slice(None, -2)
        slice3[axis] = slice(1, -1)
        slice4[axis] = slice(2, None)

        if uniform_spacing:
            out[tuple(slice1)] = (x[tuple(slice4)] - x[tuple(slice2)]) / (2.0 * ax_dx)
        else:
            dx1 = ax_dx[0:-1]
            dx2 = ax_dx[1:]
            a = -(dx2) / (dx1 * (dx1 + dx2))
            b = (dx2 - dx1) / (dx1 * dx2)
            c = dx1 / (dx2 * (dx1 + dx2))

            out[tuple(slice1)] = (
                a * x[tuple(slice2)] + b * x[tuple(slice3)] + c * x[tuple(slice4)]
            )

        # Numerical differentiation: 1st order edges
        if edge_order == 1:
            slice1[axis] = 0
            slice2[axis] = 1
            slice3[axis] = 0
            dx_0 = ax_dx if uniform_spacing else ax_dx[0]
            # 1D equivalent -- out[0] = (f[1] - f[0]) / (x[1] - x[0])
            out[tuple(slice1)] = (x[tuple(slice2)] - x[tuple(slice3)]) / dx_0

            slice1[axis] = -1
            slice2[axis] = -1
            slice3[axis] = -2
            dx_n = ax_dx if uniform_spacing else ax_dx[-1]
            # 1D equivalent -- out[-1] = (f[-1] - f[-2]) / (x[-1] - x[-2])
            out[tuple(slice1)] = (x[tuple(slice2)] - x[tuple(slice3)]) / dx_n

        # Numerical differentiation: 2nd order edges
        else:
            slice1[axis] = 0
            slice2[axis] = 0
            slice3[axis] = 1
            slice4[axis] = 2
            if uniform_spacing:
                a = -1.5 / ax_dx
                b = 2.0 / ax_dx
                c = -0.5 / ax_dx
            else:
                dx1 = ax_dx[0]
                dx2 = ax_dx[1]
                a = -(2.0 * dx1 + dx2) / (dx1 * (dx1 + dx2))
                b = (dx1 + dx2) / (dx1 * dx2)
                c = -dx1 / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[0] = a * f[0] + b * f[1] + c * f[2]
            out[tuple(slice1)] = (
                a * x[tuple(slice2)] + b * x[tuple(slice3)] + c * x[tuple(slice4)]
            )

            slice1[axis] = -1
            slice2[axis] = -3
            slice3[axis] = -2
            slice4[axis] = -1
            if uniform_spacing:
                a = 0.5 / ax_dx
                b = -2.0 / ax_dx
                c = 1.5 / ax_dx
            else:
                dx1 = ax_dx[-2]
                dx2 = ax_dx[-1]
                a = (dx2) / (dx1 * (dx1 + dx2))
                b = -(dx2 + dx1) / (dx1 * dx2)
                c = (2.0 * dx2 + dx1) / (dx2 * (dx1 + dx2))
            # 1D equivalent -- out[-1] = a * f[-3] + b * f[-2] + c * f[-1]
            out[tuple(slice1)] = (
                a * x[tuple(slice2)] + b * x[tuple(slice3)] + c * x[tuple(slice4)]
            )

        with tf.device(device):
            outvals.append(tf.convert_to_tensor(out))

        # reset the slice object in this dimension to ":"
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if len_axes == 1:
        return outvals[0]
    else:
        return outvals

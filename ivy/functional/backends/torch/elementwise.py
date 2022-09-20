# global
import torch
from typing import Union, Optional

# local
import ivy


def _cast_for_unary_op(x):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x


def add(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.add(x1, x2, out=out)


add.support_native_out = True


def bitwise_xor(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.bitwise_xor(x1, x2, out=out)


bitwise_xor.support_native_out = True


def expm1(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.expm1(x, out=out)


expm1.unsupported_dtypes = ("float16",)
expm1.support_native_out = True


def bitwise_invert(
    x: Union[int, bool, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.bitwise_not(x, out=out)


bitwise_invert.support_native_out = True


def isfinite(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.isfinite(x)


def isinf(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.isinf(x)


def equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.eq(x1, x2, out=out)


equal.support_native_out = True


def less_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.less_equal(x1, x2, out=out)


less_equal.support_native_out = True


def bitwise_and(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.bitwise_and(x1, x2, out=out)


bitwise_and.support_native_out = True


def ceil(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.ceil(x, out=out)


ceil.support_native_out = True
ceil.unsupported_dtypes = ("float16",)


def floor(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.floor(x, out=out)


floor.support_native_out = True
floor.unsupported_dtypes = ("float16",)


def asin(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.asin(x, out=out)


asin.support_native_out = True
asin.unsupported_dtypes = ("float16",)


def asinh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.asinh(x, out=out)


asinh.support_native_out = True
asinh.unsupported_dtypes = ("float16",)


def sign(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.sign(x, out=out)


sign.support_native_out = True


def sqrt(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.sqrt(x, out=out)


sqrt.support_native_out = True
sqrt.unsupported_dtypes = ("float16",)


def cosh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.cosh(x, out=out)


cosh.support_native_out = True
cosh.unsupported_dtypes = ("float16",)


def log10(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.log10(x, out=out)


log10.support_native_out = True
log10.unsupported_dtypes = ("float16",)


def log2(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.log2(x, out=out)


log2.unsupported_dtypes = ("float16",)


def log1p(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.log1p(x, out=out)


log1p.support_native_out = True
log1p.unsupported_dtypes = ("float16",)


def isnan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.isnan(x)


def less(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.lt(x1, x2, out=out)


less.support_native_out = True


def multiply(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.multiply(x1, x2, out=out)


multiply.support_native_out = True


def cos(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.cos(x, out=out)


cos.support_native_out = True
cos.unsupported_dtypes = ("float16",)


def logical_not(
    x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.logical_not(x.type(torch.bool), out=out)


logical_not.support_native_out = True


def divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret = torch.div(x1, x2)
    if ivy.is_float_dtype(x1.dtype):
        ret = ret.to(x1.dtype)
    else:
        ret = ret.to(ivy.default_float_dtype(as_native=True))
    return ret


divide.support_native_out = True


def greater(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.greater(x1, x2, out=out)


greater.support_native_out = True


def greater_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.greater_equal(x1, x2, out=out)


greater_equal.support_native_out = True


def acos(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.acos(x, out=out)


acos.support_native_out = True
acos.unsupported_dtypes = ("float16",)


def logical_xor(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.logical_xor(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_xor.support_native_out = True


def logical_and(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.logical_and(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_and.support_native_out = True


def logical_or(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return torch.logical_or(x1.type(torch.bool), x2.type(torch.bool), out=out)


logical_or.support_native_out = True


def acosh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.acosh(x, out=out)


acosh.support_native_out = True
acosh.unsupported_dtypes = ("float16",)


def sin(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.sin(x, out=out)


sin.support_native_out = True
sin.unsupported_dtypes = ("float16",)


def negative(
    x: Union[float, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.neg(x, out=out)


negative.support_native_out = True


def not_equal(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.not_equal(x1, x2, out=out)


not_equal.support_native_out = True


def tanh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.tanh(x, out=out)


tanh.support_native_out = True
tanh.unsupported_dtypes = ("float16",)


def floor_divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.div(x1, x2, rounding_mode="floor", out=out)


floor_divide.support_native_out = True


def bitwise_or(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.bitwise_or(x1, x2, out=out)


bitwise_or.support_native_out = True


def sinh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.sinh(x, out=out)


sinh.support_native_out = True
sinh.unsupported_dtypes = ("float16",)


def positive(
    x: Union[float, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.positive(x)


def square(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.square(x, out=out)


square.support_native_out = True


def pow(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.pow(x1, x2, out=out)


pow.support_native_out = True


def round(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    if "int" in str(x.dtype):
        if ivy.exists(out):
            return ivy.inplace_update(out, x)
        return x
    return torch.round(x, out=out)


round.support_native_out = True
round.unsupported_dtypes = ("float16",)


def trunc(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    if "int" not in str(x.dtype):
        return torch.trunc(x, out=out)
    ret = x
    if ivy.exists(out):
        return ivy.inplace_update(out, ret)
    return ret


trunc.support_native_out = True
trunc.unsupported_dtypes = ("float16",)


def abs(
    x: Union[float, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.abs(x, out=out)


abs.support_native_out = True


def logaddexp(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.logaddexp(x1, x2, out=out)


logaddexp.support_native_out = True
logaddexp.unsupported_dtypes = ("float16",)


def tan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.tan(x, out=out)


tan.support_native_out = True
tan.unsupported_dtypes = ("float16",)


def atan(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.atan(x, out=out)


atan.support_native_out = True
atan.unsupported_dtypes = ("float16",)


def atan2(
    x1: torch.Tensor, x2: torch.Tensor, /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.atan2(x1, x2, out=out)


atan2.support_native_out = True
atan2.unsupported_dtypes = (
    "float16",
    "bfloat16",
)  # TODO Fixed in PyTorch 1.12.1


def log(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.log(x, out=out)


log.support_native_out = True
log.unsupported_dtypes = ("float16",)


def exp(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.exp(x, out=out)


exp.support_native_out = True
exp.unsupported_dtypes = ("float16",)


def subtract(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.subtract(x1, x2, out=out)


subtract.support_native_out = True


def remainder(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    modulus: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    if not modulus:
        res = x1 / x2
        res_floored = torch.where(res >= 0, torch.floor(res), torch.ceil(res))
        diff = res - res_floored
        diff, x2 = ivy.promote_types_of_inputs(diff, x2)
        return torch.round(torch.mul(diff, x2, out=out), out=out).to(x1.dtype)
    return torch.remainder(x1, x2, out=out)


remainder.support_native_out = True
remainder.unsupported_dtypes = ("float16",)


def atanh(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.atanh(x, out=out)


atanh.support_native_out = True
atanh.unsupported_dtypes = ("float16",)


def bitwise_right_shift(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ivy.assertions.check_all(x2 >= 0, message="shifts must be non-negative")
    return torch.bitwise_right_shift(x1, x2, out=out)


bitwise_right_shift.support_native_out = True


def bitwise_left_shift(
    x1: Union[int, bool, torch.Tensor],
    x2: Union[int, bool, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ivy.assertions.check_all(x2 >= 0, message="shifts must be non-negative")
    return torch.bitwise_left_shift(x1, x2, out=out)


bitwise_left_shift.support_native_out = True


# Extra #
# ------#


def erf(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.erf(x, out=out)


erf.support_native_out = True
erf.unsupported_dtypes = ("float16",)


def minimum(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.min(x1, x2, out=out)


minimum.support_native_out = True


def maximum(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return torch.max(x1, x2, out=out)


maximum.support_native_out = True


def reciprocal(
    x: Union[float, torch.Tensor], /, *, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    x = _cast_for_unary_op(x)
    return torch.reciprocal(x, out=out)


reciprocal.support_native_out = True
reciprocal.unsupported_dtypes = ("float16",)


def deg2rad(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.deg2rad(x, out=out)


deg2rad.support_native_out = True


def rad2deg(x: torch.Tensor, /, *, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.rad2deg(x, out=out)


rad2deg.support_native_out = True


def trunc_divide(
    x1: Union[float, torch.Tensor],
    x2: Union[float, torch.Tensor],
    /,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    ret = torch.div(x1, x2, rounding_mode="trunc")
    if ivy.is_float_dtype(x1.dtype):
        ret = ret.to(x1.dtype)
    else:
        ret = ret.to(ivy.default_float_dtype(as_native=True))
    return ret

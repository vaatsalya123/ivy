# global
import ivy
import ivy.functional.frontends.torch as torch_frontend


def add(input, other, *, alpha=None, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.add(input, other, alpha=alpha, out=out)


def tan(input, *, out=None):
    return ivy.tan(input, out=out)


def atan(input, *, out=None):
    return ivy.atan(input, out=out)


def tanh(input, *, out=None):
    return ivy.tanh(input, out=out)


def cos(input, *, out=None):
    return ivy.cos(input, out=out)


def sin(input, *, out=None):
    return ivy.sin(input, out=out)


def acos(input, *, out=None):
    return ivy.acos(input, out=out)


def sinh(input, *, out=None):
    return ivy.sinh(input, out=out)


def acosh(input, *, out=None):
    return ivy.acosh(input, out=out)


def arccosh(input, *, out=None):
    return ivy.acosh(input, out=out)


def arccos(input, *, out=None):
    return ivy.acos(input, out=out)


def abs(input, *, out=None):
    return ivy.abs(input, out=out)


def cosh(input, *, out=None):
    return ivy.cosh(input, out=out)


def subtract(input, other, *, alpha=1, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.subtract(input, other * alpha, out=out)


def exp(input, *, out=None):
    return ivy.exp(input, out=out)


def asin(input, *, out=None):
    return ivy.asin(input, out=out)


def arcsin(input, *, out=None):
    return ivy.asin(input, out=out)


def asinh(input, *, out=None):
    return ivy.asinh(input, out=out)


def atanh(input, *, out=None):
    return ivy.atanh(input, out=out)


def arctanh(input, *, out=None):
    return ivy.atanh(input, out=out)


def log2(input, *, out=None):
    return ivy.log2(input, out=out)


def square(input, *, out=None):
    return ivy.square(input, out=out)


def atan2(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.atan2(input, other, out=out)


def negative(input, *, out=None):
    return ivy.negative(input, out=out)


def bitwise_and(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.bitwise_and(input, other, out=out)


def bitwise_not(input, *, out=None):
    return ivy.bitwise_invert(input, out=out)


def bitwise_xor(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.bitwise_xor(input, other, out=out)


def bitwise_or(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.bitwise_or(input, other, out=out)


def bitwise_left_shift(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.bitwise_left_shift(input, other, out=out)


def bitwise_right_shift(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.bitwise_right_shift(input, other, out=out)


def log10(input, *, out=None):
    return ivy.log10(input, out=out)


def trunc(input, *, out=None):
    return ivy.trunc(input, out=out)


def sqrt(input, *, out=None):
    return ivy.sqrt(input, out=out)


def sign(input, *, out=None):
    return ivy.sign(input, out=out)


def absolute(input, *, out=None):
    return ivy.abs(input, out=out)


def logical_not(input, *, out=None):
    return ivy.logical_not(input, out=out)


def logical_and(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.logical_and(input, other, out=out)


def logical_or(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.logical_or(input, other, out=out)


def logical_xor(input, other, *, out=None):
    input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
    return ivy.logical_xor(input, other, out=out)


def ceil(input, *, out=None):
    return ivy.ceil(input, out=out)


def clamp(input, min=None, max=None, *, out=None):
    ivy.assertions.check_all_or_any_fn(
        min,
        max,
        fn=ivy.exists,
        type="any",
        limit=[1, 2],
        message="at most one of min or max can be None",
    )
    input = ivy.array(input)
    if min is None:
        return ivy.minimum(input, max, out=out)
    if max is None:
        return ivy.maximum(input, min, out=out)
    return ivy.clip(input, min, max, out=out)


def clip(input, min=None, max=None, *, out=None):
    ivy.assertions.check_all_or_any_fn(
        min,
        max,
        fn=ivy.exists,
        type="any",
        limit=[1, 2],
        message="at most one of min or max can be None",
    )
    input = ivy.array(input)
    if min is None:
        return ivy.minimum(input, max, out=out)
    if max is None:
        return ivy.maximum(input, min, out=out)
    return ivy.clip(input, min, max, out=out)


def mul(input, other, *, out=None):
    return ivy.multiply(input, other, out=out)


multiply = mul


def div(input, other, *, rounding_mode=None, out=None):
    if rounding_mode is not None:
        input, other = torch_frontend.promote_types_of_torch_inputs(input, other)
        promoted = input.dtype
        if rounding_mode == "trunc":
            return ivy.trunc_divide(input, other, out=out).astype(promoted)
        else:
            return ivy.floor_divide(input, other, out=out).astype(promoted)
    else:
        return ivy.divide(input, other, out=out)

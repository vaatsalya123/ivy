# global
import ivy
from ivy.func_wrapper import with_unsupported_dtypes
from .. import versions


def _compute_threshold(input, threshold, value, inplace):
    ret = ivy.where(ivy.greater(input, threshold), input, value)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def _compute_elu(input, alpha=1.0, inplace=False):
    prod = ivy.multiply(
        alpha,
        ivy.subtract(ivy.exp(input), 1),
    )
    ret = ivy.where(ivy.greater(input, 0), input, prod)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def _selu_with_inplace(input, inplace=False):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    prod = ivy.multiply(
        alpha,
        ivy.subtract(
            ivy.exp(input),
            1,
        ),
    )
    min_ = ivy.multiply(
        scale,
        ivy.minimum(0, prod),
    )
    max_ = ivy.multiply(
        scale,
        ivy.maximum(0, input),
    )
    ret = ivy.add(min_, max_)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def _rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    if training:
        # alpha = ivy.random_uniform(low=lower, high=upper)
        # ToDo implement alpha correctly after fixing ivy.random_uniform
        pass
    else:
        alpha = (lower + upper) / 2
    ret = ivy.subtract(
        ivy.relu(input), ivy.multiply(alpha, ivy.relu(ivy.negative(input)))
    )
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def sigmoid(input):
    return ivy.sigmoid(input)


def leaky_relu(input, negative_slope=0.01, inplace=False):
    ret = ivy.leaky_relu(input, alpha=negative_slope)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def softmax(input, dim=None, _stacklevel=3, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(input, axis=dim)


def gelu(
    input,
):  # , *, approximate="none"): ToDo: approximate is added in in PyTorch 1.12.1
    # if approximate == "none":
    # approximate = False
    # else:
    # approximate = True
    return ivy.gelu(input, approximate=False)


def tanh(input):
    return ivy.tanh(input)


def logsigmoid(input):
    return ivy.negative(ivy.softplus(ivy.negative(input)))


def softmin(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    return ivy.softmax(-input, axis=dim)


def threshold(input, threshold, value, inplace=False):
    return _compute_threshold(input, threshold, value, inplace)


def threshold_(input, threshold, value):
    return _compute_threshold(input, threshold, value, inplace=True)


def relu6(input, inplace=False):
    ret = ivy.minimum(ivy.maximum(input, 0), 6)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def elu(input, alpha=1.0, inplace=False):
    return _compute_elu(input, alpha, inplace=inplace)


def elu_(input, alpha=1.0):
    return _compute_elu(input, alpha, inplace=True)


def celu(input, alpha=1.0, inplace=False):
    prod = ivy.multiply(
        alpha,
        ivy.subtract(
            ivy.exp(ivy.divide(input, alpha)),
            1,
        ),
    )
    ret = ivy.add(
        ivy.maximum(0, input),
        ivy.minimum(0, prod),
    )
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def selu(input, inplace=False):
    return _selu_with_inplace(input, inplace=inplace)


def prelu(input, weight):
    return ivy.add(ivy.maximum(0, input), ivy.multiply(weight, ivy.minimum(0, input)))


def rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    return _rrelu(input, lower, upper, training, inplace)


def rrelu_(input, lower=1.0 / 8, upper=1.0 / 3, training=False):
    return _rrelu(input, lower, upper, training, inplace=True)


def hardshrink(input, lambd=0.5):
    mask = ivy.logical_or(ivy.greater(input, lambd), ivy.less(input, -lambd))
    return ivy.where(mask, input, 0.0)


def softsign(input):
    return ivy.divide(input, ivy.add(1, ivy.abs(input)))


def softshrink(input, lambd=0.5):
    low = ivy.where(ivy.less(input, -lambd), ivy.add(input, lambd), 0)
    up = ivy.where(ivy.greater(input, lambd), ivy.subtract(input, lambd), 0)
    return ivy.add(low, up)


def silu(input, inplace=False):
    ret = ivy.multiply(input, ivy.sigmoid(input))
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def glu(input, dim=-1):
    a, b = ivy.split(input, num_or_size_splits=2, axis=dim)
    return ivy.multiply(a, ivy.sigmoid(b))


def log_softmax(input, dim=None, dtype=None):
    if dtype:
        input = ivy.astype(ivy.array(input), ivy.as_ivy_dtype(dtype))
    if dim is None:
        dim = -1
    return ivy.log_softmax(input, axis=dim)


def tanhshrink(input):
    return ivy.subtract(input, ivy.tanh(input))


def leaky_relu_(input, negative_slope=0.01):
    ret = ivy.leaky_relu(input, alpha=negative_slope)
    ivy.inplace_update(input, ret)
    return input


def hardswish(input, inplace=False):
    relu6_val = ivy.minimum(ivy.maximum(ivy.add(input, 3), 0), 6)
    ret = ivy.multiply(input, ivy.divide(relu6_val, 6))
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def hardsigmoid(input, inplace=False):
    ret = ivy.divide(ivy.minimum(ivy.maximum(ivy.add(input, 3), 0), 6), 6)
    if inplace:
        ivy.inplace_update(input, ret)
        return input
    return ret


def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    less = ivy.where(ivy.less(input, min_val), min_val, input)
    ret = ivy.where(ivy.greater(input, max_val), max_val, less)
    if inplace:
        return ivy.inplace_update(input, ret)
    return ret


def hardtanh_(input, min_val=-1.0, max_val=1.0):
    less = ivy.where(ivy.less(input, min_val), min_val, input)
    ret = ivy.where(ivy.greater(input, max_val), max_val, less)
    ivy.inplace_update(input, ret)
    return input


def normalize(input, p=2.0, dim=1, eps=1e-12, out=None):
    abs_square = ivy.pow(ivy.abs(input), p)
    sum_ = ivy.sum(abs_square, axis=dim, keepdims=True)
    pnorm_res = ivy.pow(sum_, 1.0 / p)
    max_ = ivy.maximum(pnorm_res, eps)
    return ivy.divide(input, max_, out=out)


@with_unsupported_dtypes({"1.11.0 and below": ("float16",)}, versions["torch"])
def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    shape = ivy.shape(input)
    if isinstance(normalized_shape, int) and normalized_shape == shape[-1]:
        axis = [-1]
    else:
        assert normalized_shape == shape[-len(normalized_shape) :]
        axis = list(range(len(shape) - len(normalized_shape), len(shape)))
    return ivy.layer_norm(input, axis, weight=weight, bias=bias, epsilon=eps)

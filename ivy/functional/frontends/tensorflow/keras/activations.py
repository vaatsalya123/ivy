import ivy


def hard_sigmoid(x):
    dtype_in = x.dtype
    point_two = ivy.full(x.shape, 0.2)
    point_five = ivy.full(x.shape, 0.5)
    x = ivy.multiply(x, point_two)
    x = ivy.add(x, point_five)
    x = ivy.clip(x, 0.0, 1.0)
    x = ivy.asarray(x, dtype=dtype_in)
    return x


def linear(x):
    return ivy.array(x)


def relu(x):
    return ivy.relu(x)


def sigmoid(x):
    return ivy.sigmoid(x)


def softmax(x, axis=-1):
    return ivy.softmax(x, axis=axis)


def gelu(x, approximate=False):
    return ivy.gelu(x, approximate=approximate)


def softplus(x):
    return ivy.softplus(x)


def softsign(x):
    return ivy.divide(x, ivy.add(1, ivy.abs(x)))


def swish(x):
    return ivy.multiply(x, ivy.sigmoid(x))


def elu(x, alpha=1.0):
    zeros = ivy.zeros_like(x, dtype=ivy.dtype(x))
    ones = ivy.ones_like(x, dtype=ivy.dtype(x))
    alpha = ivy.astype(ivy.array(alpha), ivy.dtype(x))
    ret_val = ivy.where(
        x > zeros, x, ivy.multiply(alpha, ivy.subtract(ivy.exp(x), ones))
    )
    return ret_val


elu.supported_dtypes = {
    "numpy": (
        "float16",
        "float32",
        "float64",
    ),
    "tensorflow": (
        "bfloat16",
        "float16",
        "float32",
        "float64",
    ),
    "torch": (
        "bfloat16",
        "float32",
        "float64",
    ),
    "jax": (
        "bfloat16",
        "float16",
        "float32",
        "float64",
    ),
}


def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = ivy.astype(ivy.array(1.0507009873554804934193349852946), ivy.dtype(x))
    return ivy.multiply(scale, elu(x=x, alpha=alpha))


selu.supported_dtypes = {
    "numpy": (
        "float16",
        "float32",
        "float64",
    ),
    "tensorflow": (
        "float16",
        "float32",
        "float64",
    ),
    "torch": (
        "float32",
        "float64",
    ),
    "jax": (
        "float16",
        "float32",
        "float64",
    ),
}

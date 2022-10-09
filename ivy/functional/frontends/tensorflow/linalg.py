# global
import ivy

from .. import versions
from ivy.func_wrapper import with_unsupported_dtypes

from ivy.functional.frontends.tensorflow import promote_types_of_tensorflow_inputs


def matrix_rank(a, tol=None, valiate_args=False, name=None):
    return ivy.matrix_rank(a, tol)


def det(input, name=None):
    return ivy.det(input)


def eigh(tensor, name=None):
    return ivy.eigh(tensor)


def eigvalsh(tensor, name=None):
    return ivy.eigvalsh(tensor)


def solve(x, y):
    x, y = promote_types_of_tensorflow_inputs(x, y)
    return ivy.solve(x, y)


def logdet(matrix, name=None):
    return ivy.det(matrix).log()


logdet.supported_dtypes = ("float16", "float32", "float64")


def slogdet(input, name=None):
    return ivy.slogdet(input)


def cholesky_solve(chol, rhs, name=None):
    chol, rhs = promote_types_of_tensorflow_inputs(chol, rhs)
    y = ivy.solve(chol, rhs)
    return ivy.solve(ivy.matrix_transpose(chol), y)


def pinv(a, rcond=None, validate_args=False, name=None):
    return ivy.pinv(a, rcond)


def tensordot(a, b, axes, name=None):
    a, b = promote_types_of_tensorflow_inputs(a, b)
    return ivy.tensordot(a, b, axes)


@with_unsupported_dtypes(
    {"2.9.0 and below": ("float16", "bfloat16")}, versions["tensorflow"]
)
def eye(num_rows, num_columns=None, batch_shape=None, dtype=ivy.float32, name=None):
    return ivy.eye(num_rows, num_columns, batch_shape=batch_shape, dtype=dtype)

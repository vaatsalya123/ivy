# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import to_ivy_arrays_and_back

from ivy.func_wrapper import with_unsupported_dtypes


# solve
@with_unsupported_dtypes({"1.23.0 and below": ("float16",)}, "numpy")
@to_ivy_arrays_and_back
def norm(x, ord=None, axis=None, keepdims=False):
    ret = ivy.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
    if axis is None:
        return ret[0]
    return ret


# matrix_rank
# TODO: add support for hermitian
@to_ivy_arrays_and_back
def matrix_rank(A, tol=None, hermitian=False):
    ret = ivy.matrix_rank(A, rtol=tol)
    return ivy.array(ret, dtype=ivy.int64)


# det
@to_ivy_arrays_and_back
def det(a):
    return ivy.det(a)


# slogdet
@to_ivy_arrays_and_back
def slogdet(a):
    sign, logabsdet = ivy.slogdet(a)
    return ivy.concat((ivy.reshape(sign, (-1,)), ivy.reshape(logabsdet, (-1,))))

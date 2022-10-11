# local
import ivy
from ivy.functional.frontends.numpy.func_wrapper import (
    to_ivy_arrays_and_back,
    handle_numpy_casting,
)


@handle_numpy_casting
@to_ivy_arrays_and_back
def concatenate(arrays, /, axis=0, out=None, *, dtype=None, casting="same_kind"):
    if dtype:
        arrays = [ivy.astype(ivy.array(a), ivy.as_ivy_dtype(dtype)) for a in arrays]
    return ivy.concat(arrays, axis=axis, out=out)

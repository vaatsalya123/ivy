from typing import Union, Optional, Callable, Dict
import functools
import ivy


def add_wrapper(fn: Callable):
    @functools.wraps(fn)
    def new_fn(
        x1: ivy.Array,
        x2: ivy.Array,
        alpha: float = None,
        out: ivy.Array = None,
    ):
        # inputs_to_native_arrays
        x1, x2 = x1._data, x2._data
        # outputs_to_ivy_arrays
        ret: ivy.Array = fn(x1, x2, alpha=alpha)
        return ivy.asarray(ret)
    new_fn.add_wrapper = True
    return new_fn

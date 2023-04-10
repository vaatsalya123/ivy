from typing import Union, Optional, Callable, Dict
import functools
import ivy
import cython


cdef get_array_mode():
    return ivy.get_array_mode()


cdef args_to_native(
    x1: ivy.Array,
    x2: ivy.Array,
    include_derived: Dict[type, bool] = None,
    cont_inplace: bool = False,
):
    return ivy.args_to_native(x1, x2, include_derived=include_derived, cont_inplace=cont_inplace)


cdef res_ret(ret, nested=True, include_derived=None):
    if get_array_mode():
        return ivy.to_ivy(ret, nested=nested, include_derived=include_derived)
    return ret


def add_wrapper(fn: Callable):
    @functools.wraps(fn)
    def new_fn(
        x1: ivy.Array,
        x2: ivy.Array,
        alpha: float = None,
        out: ivy.Array = None,
    ):
        # outputs_to_ivy_arrays

        # call unmodified function

        # inputs_to_native_arrays

        if not get_array_mode():
            ret: ivy.Array = fn(x1, x2, alpha=alpha, out=out)
        else:
            # check if kwargs contains an out argument, and if so, remove it
            has_out: bool = False
            out_temp = None
            if out is not None:
                out_temp = out
                has_out = True
            # convert all arrays in the inputs to ivy.NativeArray instances
            (x1, x2), _ = args_to_native(x1, x2)
            # add the original out argument back to the keyword arguments
            ret: ivy.Array = fn(x1, x2, alpha=alpha, out=out_temp)
        
        # -----------------------

        # convert all arrays in the return to `ivy.Array` instances
        return res_ret(ret, nested=True, include_derived={tuple: True})
        # -----------------------
    new_fn.add_wrapper = True
    return new_fn

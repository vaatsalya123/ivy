from typing import Union, Optional, Callable
import functools
import ivy
from ivy.func_wrapper import *
from ivy.utils.exceptions import _print_traceback_history, IvyNotImplementedException


def add_wrapper(fn: Callable):
    @functools.wraps(fn)
    def new_fn(
        x1: ivy.Array,
        x2: ivy.Array,
        alpha: Optional[float] = None,
        out: Optional[ivy.Array] = None,
    ):
        # handle_exceptions
        try:
            # handle_nestable

            if ivy.get_nestable_mode() and (
                isinstance(x1, ivy.Container)
                or isinstance(x2, ivy.Container)
            ):
                fn_name = fn.__name__
                # if any of the arguments or keyword arguments passed to the function contains
                # a container, get the container's version of the function and call it using
                # the passed arguments.
                if hasattr(ivy.Container, "_static_" + fn_name):
                    cont_fn = getattr(ivy.Container, "_static_" + fn_name)
                else:
                    # ToDo: figure out a way to make sure that fn here is the function after the rest of the wrappers after handle_nestable
                    cont_fn = lambda x1, x2, alpha=None, out=None: ivy.Container.cont_multi_map_in_function(
                        fn, x1, x2, alpha=alpha, out=out
                    )
                ret = cont_fn(x1, x2, alpha=alpha, out=out)
            else:
                array_mode: bool = ivy.get_array_mode()
                if not array_mode:
                    ret = fn(x1, x2, alpha=alpha, out=out)
                    return ret
                # if the passed arguments does not contain a container, the function using
                # the passed arguments, returning an ivy or a native array.

                # handle_out_argument

                if out is None:

                    # to_native_arrays_and_back

                    # outputs_to_ivy_arrays
                    
                    # call unmodified function

                    # inputs_to_native_arrays

                    # check if kwargs contains an out argument, and if so, remove it
                    has_out = False
                    out_temp = None
                    if out is not None:
                        out_temp = out
                        has_out = True
                    # convert all arrays in the inputs to ivy.NativeArray instances
                    x1, x2 = x1._data if isinstance(x1, ivy.Array) else x1, x2._data if isinstance(x2, ivy.Array) else x2
                    # add the original out argument back to the keyword arguments
                    if has_out:
                        out = out_temp
                    ret = fn(x1, x2, alpha=alpha, out=out)

                    # --------------------------------------

                    # convert all arrays in the return to `ivy.Array` instances
                    ret = ivy.Array(ret)

                    # --------------------------------------

                    # --------------------------------------

                else:
                    handle_out_in_backend = hasattr(fn, "support_native_out")
                    if handle_out_in_backend:
                        # extract underlying native array for out
                        native_out = out._data
                        # compute return, with backend inplace update handled by
                        # the backend function


                        # to_native_arrays_and_back

                        # outputs_to_ivy_arrays

                        # call unmodified function

                        # inputs_to_native_arrays

                        # check if kwargs contains an out argument, and if so, remove it
                        has_out = False
                        out_temp = None
                        if out is not None:
                            out_temp = native_out
                            has_out = True
                        # convert all arrays in the inputs to ivy.NativeArray instances
                        x1, x2 = x1._data if isinstance(x1, ivy.Array) else x1, x2._data if isinstance(x2, ivy.Array) else x2
                        # add the original out argument back to the keyword arguments
                        if has_out:
                            native_out = out_temp
                        ret = fn(x1, x2, alpha=alpha, out=native_out)

                        # --------------------------------------

                        # convert all arrays in the return to `ivy.Array` instances
                        ret = ivy.Array(ret)

                        # --------------------------------------
                        
                        # --------------------------------------

                        if isinstance(ret, (tuple, list)):
                            for i in range(len(ret)):
                                out[i]._data = ret[i]._data
                        else:
                            out._data = ret._data
                        ret = out
                    else:
                        # compute return, and then handle the inplace update explicitly


                        # to_native_arrays_and_back
                        
                        # outputs_to_ivy_arrays

                        # call unmodified function

                        # inputs_to_native_arrays
                        
                        # check if kwargs contains an out argument, and if so, remove it
                        has_out = False
                        out_temp = None
                        if out is not None:
                            out_temp = out
                            has_out = True
                        # convert all arrays in the inputs to ivy.NativeArray instances
                        x1, x2 = x1._data if isinstance(x1, ivy.Array) else x1, x2._data if isinstance(x2, ivy.Array) else x2
                        # add the original out argument back to the keyword arguments
                        if has_out:
                            out = out_temp
                        ret = fn(x1, x2, alpha=alpha, out=out)

                        # --------------------------------------

                        # convert all arrays in the return to `ivy.Array` instances
                        ret = ivy.Array(ret)

                        # --------------------------------------

                        # --------------------------------------

                        if not (isinstance(ret, ivy.Array) and ivy.is_native_array(ret.data, exclusive=exclusive) or ivy.is_native_array(ret, exclusive=exclusive)) and not isinstance(ret, ivy.Container):
                            ret = ivy.nested_multi_map(
                                lambda x, _: ivy.inplace_update(
                                    x[0], ivy.astype(x[1], ivy.dtype(x[0]))
                                ),
                                [out, ret],
                            )
                        else:
                            ret = ivy.inplace_update(out, ivy.astype(ret, ivy.dtype(out)))
                            # return output matches the dtype of the out array to match numpy and torch

                # --------------------------------------

            # --------------------------------------
            return ret
        # Not to rethrow as IvyBackendException
        except IvyNotImplementedException as e:
            raise e
        except (IndexError, ValueError, AttributeError) as e:
            _print_traceback_history()
            raise ivy.utils.exceptions.IvyError(fn.__name__, str(e))
        except Exception as e:
            _print_traceback_history()
            raise ivy.utils.exceptions.IvyBackendException(fn.__name__, str(e))

        # --------------------------------------
    new_fn.add_wrapper = True
    return new_fn

from typing import Union, Optional, Callable
import functools
import ivy
from ivy.utils.exceptions import IvyNotImplementedException
import sys
import traceback as tb


cdef get_array_mode():
    array_mode_stack: list = ivy.array_mode_stack
    array_mode: bool = True
    if array_mode_stack:
        array_mode = array_mode_stack[-1]
    return array_mode


cdef get_nestable_mode():
    nestable_mode_stack: list = ivy.nestable_mode_stack
    nestable_mode: bool = True
    if nestable_mode_stack:
        nestable_mode = nestable_mode_stack[-1]
    return nestable_mode


cdef get_native(x):
    if isinstance(x, ivy.Array):
        return x._data
    return x


cdef _log_stack_trace_truncated(trace_mode, func_wrapper_trace_mode):
    if trace_mode in ["frontend", "ivy"]:
        print(
            "<stack trace is truncated to {} specific files,".format(trace_mode),
            "call `ivy.set_exception_trace_mode('full')` to view the full trace>",
        )
    if not func_wrapper_trace_mode:
        print(
            "<func_wrapper.py stack trace is squashed,",
            "call `ivy.set_show_func_wrapper_trace_mode(True)` in order to view this>",
        )


cdef _print_new_stack_trace(old_stack_trace, trace_mode, func_wrapper_trace_mode):
    _log_stack_trace_truncated(trace_mode, func_wrapper_trace_mode)
    new_stack_trace = []
    for st in old_stack_trace:
        if trace_mode == "full" and not func_wrapper_trace_mode:
            if "func_wrapper.py" not in repr(st):
                new_stack_trace.append(st)
        else:
            if ivy.trace_mode_dict[trace_mode] in repr(st):
                if not func_wrapper_trace_mode and "func_wrapper.py" in repr(st):
                    continue
                new_stack_trace.append(st)
    print("".join(tb.format_list(new_stack_trace)))


cdef _print_traceback_history():
    trace_mode = ivy.get_exception_trace_mode()
    func_wrapper_trace_mode = ivy.get_show_func_wrapper_trace_mode()
    if trace_mode == "none":
        return
    if trace_mode == "full" and func_wrapper_trace_mode:
        print("".join(tb.format_tb(sys.exc_info()[2])))
    else:
        _print_new_stack_trace(
            tb.extract_tb(sys.exc_info()[2]), trace_mode, func_wrapper_trace_mode
        )
    print("During the handling of the above exception, another exception occurred:\n")


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
            nestable_mode: bool = get_nestable_mode()
            if nestable_mode and (
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
                array_mode: bool = get_array_mode()
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

                    # convert all arrays in the inputs to ivy.NativeArray instances
                    x1, x2 = get_native(x1), get_native(x2)
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
                        x1, x2 = get_native(x1), get_native(x2)
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
                        x1, x2 = get_native(x1), get_native(x2)
                        # add the original out argument back to the keyword arguments
                        if has_out:
                            out = out_temp
                        ret = fn(x1, x2, alpha=alpha, out=out)

                        # --------------------------------------

                        # convert all arrays in the return to `ivy.Array` instances
                        ret = ivy.Array(ret)

                        # --------------------------------------

                        # --------------------------------------

                        if not (isinstance(ret, ivy.Array) and ivy.is_native_array(ret.data) or ivy.is_native_array(ret)) and not isinstance(ret, ivy.Container):
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

import contextlib
import numbers
import ivy
import functools
import logging
import weakref
import warnings
import copy as python_copy
from types import FunctionType
from typing import Callable
import inspect
import functools
import sys
import traceback as tb



def try_array_function_override(func, overloaded_args, types, args, kwargs):
    if not overloaded_args:
        return False, None

    for overloaded_arg in overloaded_args:
        # Note that we're only calling __ivy_array_function__ on the *first*
        # occurence of each argument type. This is necessary for reasonable
        # performance with a possibly long list of overloaded arguments, for
        # which each __ivy_array_function__ implementation might reasonably need to
        # check all argument types.
        try:
            result = overloaded_arg.__ivy_array_function__(func, types, args, kwargs)
        except Exception:
            raise ivy.utils.exceptions.IvyNotImplementedException

        if result is not NotImplemented:
            return True, result

    raise TypeError(
        "no implementation found for {} on types that implement "
        "__ivy_array_function__: {}".format(func, list(map(type, overloaded_args)))
    )


def _log_stack_trace_truncated(trace_mode, func_wrapper_trace_mode):
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


def _print_new_stack_trace(old_stack_trace, trace_mode, func_wrapper_trace_mode):
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


def _print_traceback_history():
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


class IvyException(Exception):
    def __init__(self, message):
        super().__init__(message)


class IvyBackendException(IvyException):
    def __init__(self, *messages):
        self._default = [
            "numpy" if ivy.current_backend_str() == "" else ivy.current_backend_str()
        ]
        self._delimiter = ": "
        for message in messages:
            self._default.append(message)
        super().__init__(self._delimiter.join(self._default))


class IvyNotImplementedException(NotImplementedError):
    def __init__(self, message=""):
        super().__init__(message)


class IvyError(IndexError, ValueError, AttributeError, IvyException):
    def __init__(self, *messages):
        self._default = [
            "numpy" if ivy.current_backend_str() == "" else ivy.current_backend_str()
        ]
        self._delimiter = ": "
        for message in messages:
            self._default.append(message)
        super().__init__(self._delimiter.join(self._default))


def add_wrapper(fn):
    @functools.wraps(fn)
    def new_fn(*args, **kwargs):

        # handle_exceptions
        try:

            # handle_nestable
            fn_name = fn.__name__
            # if any of the arguments or keyword arguments passed to the function contains
            # a container, get the container's version of the function and call it using
            # the passed arguments.
            if hasattr(ivy.Container, "static_" + fn_name):
                cont_fn = getattr(ivy.Container, "static_" + fn_name)
            else:
                cont_fn = lambda *args, **kwargs: ivy.Container.cont_multi_map_in_function(
                    fn, *args, **kwargs     # come back to this later
                )
            if ivy.get_nestable_mode() and (
                ivy.nested_any(args, ivy.is_ivy_container, check_nests=True)
                or ivy.nested_any(kwargs, ivy.is_ivy_container, check_nests=True)
            ):
                return cont_fn(*args, **kwargs)

            # if the passed arguments does not contain a container, the function using
            # the passed arguments, returning an ivy or a native array.

            # handle_out_argument
            handle_out_in_backend = hasattr(fn, "support_native_out")
            out = None if 'out' not in kwargs else kwargs['out']
            if out is None:

                # inputs_to_native_arrays
                if not ivy.get_array_mode():

                    # outputs_to_ivy_arrays
                    # call unmodified function

                    # handle_array_function
                    overloaded_types = []
                    overloaded_args = []

                    for arg in args + tuple(kwargs.values()):
                        if ivy.exists(arg) and (
                            not isinstance(arg, ivy.Container)
                            and hasattr(arg, "__ivy_array_function__")
                        ):
                            if type(arg) not in overloaded_types:
                                overloaded_types.append(type(arg))
                                if (
                                    arg.__ivy_array_function__
                                    is not ivy.Array.__ivy_array_function__
                                    and not isinstance(arg, (ivy.Array, ivy.NativeArray))
                                ):
                                    index = len(overloaded_args)
                                    for i, old_arg in enumerate(overloaded_args):
                                        if issubclass(type(arg), type(old_arg)):
                                            index = i
                                            break
                                    overloaded_args.insert(index, arg)
                        if ivy.exists(arg) and isinstance(arg, ivy.Container):
                            arg = ivy.Container.cont_flatten_key_chains(arg)
                            indices = ivy.nested_argwhere(
                                arg, lambda x: hasattr(x, "__ivy_array_function__")
                            )
                            for a in indices:
                                if type(getattr(arg, a[0])) not in overloaded_types:
                                    overloaded_types.append(type(getattr(arg, a[0])))

                                    if getattr(
                                        arg, a[0]
                                    ).__ivy_array_function__ is not ivy.Array.__ivy_array_function__ and not isinstance(  # noqa: E501
                                        getattr(arg, a[0]), (ivy.Array, ivy.NativeArray)
                                    ):
                                        index = len(overloaded_args)
                                        for i, old_arg in enumerate(overloaded_args):
                                            if issubclass(type(getattr(arg, a[0])), type(old_arg)):
                                                index = i
                                                break
                                        overloaded_args.insert(index, arg)

                    success, value = try_array_function_override(
                        ivy.__dict__[fn.__name__], overloaded_args, overloaded_types, args, kwargs
                    )
                    if success:
                        ret = value
                    else:
                        ret = fn(*args, **kwargs)
                    # --------------------------------

                    # convert all arrays in the return to `ivy.Array` instances
                    return (
                        ivy.to_ivy(ret, nested=True, include_derived={tuple: True})
                        if ivy.get_array_mode()
                        else ret
                    )
                    # --------------------------------

                # check if kwargs contains an out argument, and if so, remove it
                has_out = False
                out = None
                if "out" in kwargs:
                    out = kwargs["out"]
                    del kwargs["out"]
                    has_out = True
                # convert all arrays in the inputs to ivy.NativeArray instances
                new_args, new_kwargs = ivy.args_to_native(*args, **kwargs)
                # add the original out argument back to the keyword arguments
                if has_out:
                    new_kwargs["out"] = out
                
                # outputs_to_ivy_arrays
                # call unmodified function

                # handle_array_function
                overloaded_types = []
                overloaded_args = []

                for arg in new_args + tuple(new_kwargs.values()):
                    if ivy.exists(arg) and (
                        not isinstance(arg, ivy.Container)
                        and hasattr(arg, "__ivy_array_function__")
                    ):
                        if type(arg) not in overloaded_types:
                            overloaded_types.append(type(arg))
                            if (
                                arg.__ivy_array_function__
                                is not ivy.Array.__ivy_array_function__
                                and not isinstance(arg, (ivy.Array, ivy.NativeArray))
                            ):
                                index = len(overloaded_args)
                                for i, old_arg in enumerate(overloaded_args):
                                    if issubclass(type(arg), type(old_arg)):
                                        index = i
                                        break
                                overloaded_args.insert(index, arg)
                    if ivy.exists(arg) and isinstance(arg, ivy.Container):
                        arg = ivy.Container.cont_flatten_key_chains(arg)
                        indices = ivy.nested_argwhere(
                            arg, lambda x: hasattr(x, "__ivy_array_function__")
                        )
                        for a in indices:
                            if type(getattr(arg, a[0])) not in overloaded_types:
                                overloaded_types.append(type(getattr(arg, a[0])))

                                if getattr(
                                    arg, a[0]
                                ).__ivy_array_function__ is not ivy.Array.__ivy_array_function__ and not isinstance(  # noqa: E501
                                    getattr(arg, a[0]), (ivy.Array, ivy.NativeArray)
                                ):
                                    index = len(overloaded_args)
                                    for i, old_arg in enumerate(overloaded_args):
                                        if issubclass(type(getattr(arg, a[0])), type(old_arg)):
                                            index = i
                                            break
                                    overloaded_args.insert(index, arg)

                success, value = try_array_function_override(
                    ivy.__dict__[fn.__name__], overloaded_args, overloaded_types, new_args, new_kwargs
                )
                if success:
                    ret = value
                else:
                    ret = fn(*new_args, **new_kwargs)
                # --------------------------------

                # convert all arrays in the return to `ivy.Array` instances
                return (
                    ivy.to_ivy(ret, nested=True, include_derived={tuple: True})
                    if ivy.get_array_mode()
                    else ret
                )
                # --------------------------------

                # --------------------------------

            kwargs.pop('out')
            if handle_out_in_backend:
                # extract underlying native array for out
                native_out = ivy.to_native(out)
                # compute return, with backend inplace update handled by
                # the backend function
                kwargs.update({'out': native_out})
                
                # inputs_to_native_arrays
                if not ivy.get_array_mode():

                    # outputs_to_ivy_arrays
                    # call unmodified function

                    # handle_array_function
                    overloaded_types = []
                    overloaded_args = []

                    for arg in args + tuple(kwargs.values()):
                        if ivy.exists(arg) and (
                            not isinstance(arg, ivy.Container)
                            and hasattr(arg, "__ivy_array_function__")
                        ):
                            if type(arg) not in overloaded_types:
                                overloaded_types.append(type(arg))
                                if (
                                    arg.__ivy_array_function__
                                    is not ivy.Array.__ivy_array_function__
                                    and not isinstance(arg, (ivy.Array, ivy.NativeArray))
                                ):
                                    index = len(overloaded_args)
                                    for i, old_arg in enumerate(overloaded_args):
                                        if issubclass(type(arg), type(old_arg)):
                                            index = i
                                            break
                                    overloaded_args.insert(index, arg)
                        if ivy.exists(arg) and isinstance(arg, ivy.Container):
                            arg = ivy.Container.cont_flatten_key_chains(arg)
                            indices = ivy.nested_argwhere(
                                arg, lambda x: hasattr(x, "__ivy_array_function__")
                            )
                            for a in indices:
                                if type(getattr(arg, a[0])) not in overloaded_types:
                                    overloaded_types.append(type(getattr(arg, a[0])))

                                    if getattr(
                                        arg, a[0]
                                    ).__ivy_array_function__ is not ivy.Array.__ivy_array_function__ and not isinstance(  # noqa: E501
                                        getattr(arg, a[0]), (ivy.Array, ivy.NativeArray)
                                    ):
                                        index = len(overloaded_args)
                                        for i, old_arg in enumerate(overloaded_args):
                                            if issubclass(type(getattr(arg, a[0])), type(old_arg)):
                                                index = i
                                                break
                                        overloaded_args.insert(index, arg)

                    success, value = try_array_function_override(
                        ivy.__dict__[fn.__name__], overloaded_args, overloaded_types, args, kwargs
                    )
                    if success:
                        ret = value
                    else:
                        ret = fn(*args, **kwargs)
                    # --------------------------------

                    # convert all arrays in the return to `ivy.Array` instances
                    ret = (
                        ivy.to_ivy(ret, nested=True, include_derived={tuple: True})
                        if ivy.get_array_mode()
                        else ret
                    )
                    # --------------------------------

                # check if kwargs contains an out argument, and if so, remove it
                has_out = False
                out = None
                if "out" in kwargs:
                    out = kwargs["out"]
                    del kwargs["out"]
                    has_out = True
                # convert all arrays in the inputs to ivy.NativeArray instances
                new_args, new_kwargs = ivy.args_to_native(*args, **kwargs)
                # add the original out argument back to the keyword arguments
                if has_out:
                    new_kwargs["out"] = out
                
                # outputs_to_ivy_arrays
                # call unmodified function

                # handle_array_function
                overloaded_types = []
                overloaded_args = []

                for arg in new_args + tuple(new_kwargs.values()):
                    if ivy.exists(arg) and (
                        not isinstance(arg, ivy.Container)
                        and hasattr(arg, "__ivy_array_function__")
                    ):
                        if type(arg) not in overloaded_types:
                            overloaded_types.append(type(arg))
                            if (
                                arg.__ivy_array_function__
                                is not ivy.Array.__ivy_array_function__
                                and not isinstance(arg, (ivy.Array, ivy.NativeArray))
                            ):
                                index = len(overloaded_args)
                                for i, old_arg in enumerate(overloaded_args):
                                    if issubclass(type(arg), type(old_arg)):
                                        index = i
                                        break
                                overloaded_args.insert(index, arg)
                    if ivy.exists(arg) and isinstance(arg, ivy.Container):
                        arg = ivy.Container.cont_flatten_key_chains(arg)
                        indices = ivy.nested_argwhere(
                            arg, lambda x: hasattr(x, "__ivy_array_function__")
                        )
                        for a in indices:
                            if type(getattr(arg, a[0])) not in overloaded_types:
                                overloaded_types.append(type(getattr(arg, a[0])))

                                if getattr(
                                    arg, a[0]
                                ).__ivy_array_function__ is not ivy.Array.__ivy_array_function__ and not isinstance(  # noqa: E501
                                    getattr(arg, a[0]), (ivy.Array, ivy.NativeArray)
                                ):
                                    index = len(overloaded_args)
                                    for i, old_arg in enumerate(overloaded_args):
                                        if issubclass(type(getattr(arg, a[0])), type(old_arg)):
                                            index = i
                                            break
                                    overloaded_args.insert(index, arg)

                success, value = try_array_function_override(
                    ivy.__dict__[fn.__name__], overloaded_args, overloaded_types, new_args, new_kwargs
                )
                if success:
                    ret = value
                else:
                    ret = fn(*new_args, **new_kwargs)
                # --------------------------------

                # convert all arrays in the return to `ivy.Array` instances
                ret = (
                    ivy.to_ivy(ret, nested=True, include_derived={tuple: True})
                    if ivy.get_array_mode()
                    else ret
                )
                # --------------------------------

                # --------------------------------

                if isinstance(ret, (tuple, list)):
                    for i in range(len(ret)):
                        out[i].data = ivy.to_native(ret[i])
                else:
                    out.data = ivy.to_native(ret)
                return out
            # compute return, and then handle the inplace update explicitly

            # inputs_to_native_arrays
            if not ivy.get_array_mode():

                # outputs_to_ivy_arrays
                # call unmodified function

                # handle_array_function
                overloaded_types = []
                overloaded_args = []

                for arg in args + tuple(kwargs.values()):
                    if ivy.exists(arg) and (
                        not isinstance(arg, ivy.Container)
                        and hasattr(arg, "__ivy_array_function__")
                    ):
                        if type(arg) not in overloaded_types:
                            overloaded_types.append(type(arg))
                            if (
                                arg.__ivy_array_function__
                                is not ivy.Array.__ivy_array_function__
                                and not isinstance(arg, (ivy.Array, ivy.NativeArray))
                            ):
                                index = len(overloaded_args)
                                for i, old_arg in enumerate(overloaded_args):
                                    if issubclass(type(arg), type(old_arg)):
                                        index = i
                                        break
                                overloaded_args.insert(index, arg)
                    if ivy.exists(arg) and isinstance(arg, ivy.Container):
                        arg = ivy.Container.cont_flatten_key_chains(arg)
                        indices = ivy.nested_argwhere(
                            arg, lambda x: hasattr(x, "__ivy_array_function__")
                        )
                        for a in indices:
                            if type(getattr(arg, a[0])) not in overloaded_types:
                                overloaded_types.append(type(getattr(arg, a[0])))

                                if getattr(
                                    arg, a[0]
                                ).__ivy_array_function__ is not ivy.Array.__ivy_array_function__ and not isinstance(  # noqa: E501
                                    getattr(arg, a[0]), (ivy.Array, ivy.NativeArray)
                                ):
                                    index = len(overloaded_args)
                                    for i, old_arg in enumerate(overloaded_args):
                                        if issubclass(type(getattr(arg, a[0])), type(old_arg)):
                                            index = i
                                            break
                                    overloaded_args.insert(index, arg)

                success, value = try_array_function_override(
                    ivy.__dict__[fn.__name__], overloaded_args, overloaded_types, args, kwargs
                )
                if success:
                    ret = value
                else:
                    ret = fn(*args, **kwargs)
                # --------------------------------

                # convert all arrays in the return to `ivy.Array` instances
                ret = (
                    ivy.to_ivy(ret, nested=True, include_derived={tuple: True})
                    if ivy.get_array_mode()
                    else ret
                )
                # --------------------------------

            # check if kwargs contains an out argument, and if so, remove it
            has_out = False
            out = None
            if "out" in kwargs:
                out = kwargs["out"]
                del kwargs["out"]
                has_out = True
            # convert all arrays in the inputs to ivy.NativeArray instances
            new_args, new_kwargs = ivy.args_to_native(*args, **kwargs)
            # add the original out argument back to the keyword arguments
            if has_out:
                new_kwargs["out"] = out
            
            # outputs_to_ivy_arrays
            # call unmodified function

            # handle_array_function
            overloaded_types = []
            overloaded_args = []

            for arg in new_args + tuple(new_kwargs.values()):
                if ivy.exists(arg) and (
                    not isinstance(arg, ivy.Container)
                    and hasattr(arg, "__ivy_array_function__")
                ):
                    if type(arg) not in overloaded_types:
                        overloaded_types.append(type(arg))
                        if (
                            arg.__ivy_array_function__
                            is not ivy.Array.__ivy_array_function__
                            and not isinstance(arg, (ivy.Array, ivy.NativeArray))
                        ):
                            index = len(overloaded_args)
                            for i, old_arg in enumerate(overloaded_args):
                                if issubclass(type(arg), type(old_arg)):
                                    index = i
                                    break
                            overloaded_args.insert(index, arg)
                if ivy.exists(arg) and isinstance(arg, ivy.Container):
                    arg = ivy.Container.cont_flatten_key_chains(arg)
                    indices = ivy.nested_argwhere(
                        arg, lambda x: hasattr(x, "__ivy_array_function__")
                    )
                    for a in indices:
                        if type(getattr(arg, a[0])) not in overloaded_types:
                            overloaded_types.append(type(getattr(arg, a[0])))

                            if getattr(
                                arg, a[0]
                            ).__ivy_array_function__ is not ivy.Array.__ivy_array_function__ and not isinstance(  # noqa: E501
                                getattr(arg, a[0]), (ivy.Array, ivy.NativeArray)
                            ):
                                index = len(overloaded_args)
                                for i, old_arg in enumerate(overloaded_args):
                                    if issubclass(type(getattr(arg, a[0])), type(old_arg)):
                                        index = i
                                        break
                                overloaded_args.insert(index, arg)

            success, value = try_array_function_override(
                ivy.__dict__[fn.__name__], overloaded_args, overloaded_types, new_args, new_kwargs
            )
            if success:
                ret = value
            else:
                ret = fn(*new_args, **new_kwargs)
            # --------------------------------

            # convert all arrays in the return to `ivy.Array` instances
            ret = (
                ivy.to_ivy(ret, nested=True, include_derived={tuple: True})
                if ivy.get_array_mode()
                else ret
            )
            # --------------------------------

            # --------------------------------

            if not ivy.is_array(ret) and not ivy.is_ivy_container(ret):
                return ivy.nested_multi_map(
                    lambda x, _: ivy.inplace_update(
                        x[0], ivy.astype(x[1], ivy.dtype(x[0]))
                    ),
                    [out, ret],
                )
            return ivy.inplace_update(out, ivy.astype(ret, ivy.dtype(out)))
            # return output matches the dtype of the out array to match numpy and torch
            # --------------------------------

            # --------------------------------

        # Not to rethrow as IvyBackendException
        except IvyNotImplementedException as e:
            raise e
        except (IndexError, ValueError, AttributeError) as e:
            _print_traceback_history()
            raise ivy.utils.exceptions.IvyError(fn.__name__, str(e))
        except Exception as e:
            _print_traceback_history()
            raise ivy.utils.exceptions.IvyBackendException(fn.__name__, str(e))
        # --------------------------------
    new_fn.add_wrapper = True
    return new_fn
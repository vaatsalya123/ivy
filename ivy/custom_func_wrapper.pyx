from typing import Union, Optional, Callable
import functools
import ivy
from ivy.func_wrapper import *


def add_wrapper(fn: Callable):
    @functools.wraps(fn)
    def new_fn(
        x1: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        x2: Union[float, ivy.Array, ivy.NativeArray, ivy.Container],
        alpha: Optional[Union[int, float, ivy.Container]] = None,
        out: Optional[Union[ivy.Array, ivy.Container]] = None,
    ):
        args = [x1, x2]
        kwargs = {'alpha': alpha, 'out': out}
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
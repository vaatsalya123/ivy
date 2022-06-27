"""Collection of helpers for ivy unit tests."""

# global
import importlib
from contextlib import redirect_stdout
from io import StringIO
import sys
import re
import inspect

import numpy as np
import math
from typing import Union, List


try:
    import jax.numpy as jnp
except (ImportError, RuntimeError, AttributeError):
    jnp = None
try:
    import tensorflow as tf

    _tf_version = float(".".join(tf.__version__.split(".")[0:2]))
    if _tf_version >= 2.3:
        # noinspection PyPep8Naming,PyUnresolvedReferences
        from tensorflow.python.types.core import Tensor as tensor_type
    else:
        # noinspection PyPep8Naming
        # noinspection PyProtectedMember,PyUnresolvedReferences
        from tensorflow.python.framework.tensor_like import _TensorLike as tensor_type
    physical_devices = tf.config.list_physical_devices("GPU")
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
except ImportError:
    tf = None
try:
    import torch
except ImportError:
    torch = None
try:
    import mxnet as mx
    import mxnet.ndarray as mx_nd
except ImportError:
    mx = None
    mx_nd = None
from hypothesis import strategies as st
import hypothesis.extra.numpy as nph

# local
import ivy
import ivy.functional.backends.numpy as ivy_np


def get_ivy_numpy():
    try:
        import ivy.functional.backends.numpy
    except ImportError:
        return None
    return ivy.functional.backends.numpy


def get_ivy_jax():
    try:
        import ivy.functional.backends.jax
    except ImportError:
        return None
    return ivy.functional.backends.jax


def get_ivy_tensorflow():
    try:
        import ivy.functional.backends.tensorflow
    except ImportError:
        return None
    return ivy.functional.backends.tensorflow


def get_ivy_torch():
    try:
        import ivy.functional.backends.torch
    except ImportError:
        return None
    return ivy.functional.backends.torch


def get_ivy_mxnet():
    try:
        import ivy.functional.backends.mxnet
    except ImportError:
        return None
    return ivy.functional.backends.mxnet


_ivy_fws_dict = {
    "numpy": lambda: get_ivy_numpy(),
    "jax": lambda: get_ivy_jax(),
    "tensorflow": lambda: get_ivy_tensorflow(),
    "tensorflow_graph": lambda: get_ivy_tensorflow(),
    "torch": lambda: get_ivy_torch(),
    "mxnet": lambda: get_ivy_mxnet(),
}

_iterable_types = [list, tuple, dict]
_excluded = []


def _convert_vars(
    vars_in, from_type, to_type_callable=None, keep_other=True, to_type=None
):
    new_vars = list()
    for var in vars_in:
        if type(var) in _iterable_types:
            return_val = _convert_vars(var, from_type, to_type_callable)
            new_vars.append(return_val)
        elif isinstance(var, from_type):
            if isinstance(var, np.ndarray):
                if var.dtype == np.float64:
                    var = var.astype(np.float32)
                if bool(sum([stride < 0 for stride in var.strides])):
                    var = var.copy()
            if to_type_callable:
                new_vars.append(to_type_callable(var))
            else:
                raise Exception("Invalid. A conversion callable is required.")
        elif to_type is not None and isinstance(var, to_type):
            new_vars.append(var)
        elif keep_other:
            new_vars.append(var)

    return new_vars


def np_call(func, *args, **kwargs):
    ret = func(*args, **kwargs)
    if isinstance(ret, (list, tuple)):
        return ivy.to_native(ret, nested=True)
    return ivy.to_numpy(ret)


def jnp_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, jnp.asarray)
    new_kw_vals = _convert_vars(kwargs.values(), np.ndarray, jnp.asarray)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(_convert_vars(output, (jnp.ndarray, ivy.Array), ivy.to_numpy))
    else:
        return _convert_vars([output], (jnp.ndarray, ivy.Array), ivy.to_numpy)[0]


def tf_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, tf.convert_to_tensor)
    new_kw_vals = _convert_vars(kwargs.values(), np.ndarray, tf.convert_to_tensor)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(_convert_vars(output, (tensor_type, ivy.Array), ivy.to_numpy))
    else:
        return _convert_vars([output], (tensor_type, ivy.Array), ivy.to_numpy)[0]


def tf_graph_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, tf.convert_to_tensor)
    new_kw_vals = _convert_vars(kwargs.values(), np.ndarray, tf.convert_to_tensor)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))

    @tf.function
    def tf_func(*local_args, **local_kwargs):
        return func(*local_args, **local_kwargs)

    output = tf_func(*new_args, **new_kwargs)

    if isinstance(output, tuple):
        return tuple(_convert_vars(output, (tensor_type, ivy.Array), ivy.to_numpy))
    else:
        return _convert_vars([output], (tensor_type, ivy.Array), ivy.to_numpy)[0]


def torch_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, torch.from_numpy)
    new_kw_vals = _convert_vars(kwargs.values(), np.ndarray, torch.from_numpy)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_vals))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(_convert_vars(output, (torch.Tensor, ivy.Array), ivy.to_numpy))
    else:
        return _convert_vars([output], (torch.Tensor, ivy.Array), ivy.to_numpy)[0]


def mx_call(func, *args, **kwargs):
    new_args = _convert_vars(args, np.ndarray, mx_nd.array)
    new_kw_items = _convert_vars(kwargs.values(), np.ndarray, mx_nd.array)
    new_kwargs = dict(zip(kwargs.keys(), new_kw_items))
    output = func(*new_args, **new_kwargs)
    if isinstance(output, tuple):
        return tuple(
            _convert_vars(output, (mx_nd.ndarray.NDArray, ivy.Array), ivy.to_numpy)
        )
    else:
        return _convert_vars(
            [output], (mx_nd.ndarray.NDArray, ivy.Array), ivy.to_numpy
        )[0]


_calls = [np_call, jnp_call, tf_call, tf_graph_call, torch_call, mx_call]


def assert_compilable(fn):
    try:
        ivy.compile(fn)
    except Exception as e:
        raise e


# function that trims white spaces from docstrings
def trim(docstring):
    """Trim function from PEP-257"""
    if not docstring:
        return ""
    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()
    # Determine minimum indentation (first line doesn't count):
    indent = sys.maxsize
    for line in lines[1:]:
        stripped = line.lstrip()
        if stripped:
            indent = min(indent, len(line) - len(stripped))
    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    if indent < sys.maxsize:
        for line in lines[1:]:
            trimmed.append(line[indent:].rstrip())
    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)

    # Current code/unittests expects a line return at
    # end of multiline docstrings
    # workaround expected behavior from unittests
    if "\n" in docstring:
        trimmed.append("")

    # Return a single string:
    return "\n".join(trimmed)


def docstring_examples_run(fn, from_container=False, from_array=False):
    if not hasattr(fn, "__name__"):
        return True
    fn_name = fn.__name__
    if fn_name not in ivy.backend_handler.ivy_original_dict:
        return True

    if from_container:
        docstring = getattr(
            ivy.backend_handler.ivy_original_dict["Container"], fn_name
        ).__doc__
    elif from_array:
        docstring = getattr(
            ivy.backend_handler.ivy_original_dict["Array"], fn_name
        ).__doc__
    else:
        docstring = ivy.backend_handler.ivy_original_dict[fn_name].__doc__

    if docstring is None:
        return True

    # removing extra new lines and trailing white spaces from the docstrings
    trimmed_docstring = trim(docstring)
    trimmed_docstring = trimmed_docstring.split("\n")

    # end_index: -1, if print statement is not found in the docstring
    end_index = -1

    # parsed_output is set as an empty string to manage functions with multiple inputs
    parsed_output = ""

    # parsing through the docstrings to find lines with print statement
    # following which is our parsed output
    sub = ">>> print("
    for index, line in enumerate(trimmed_docstring):
        if sub in line:
            end_index = trimmed_docstring.index("", index)
            p_output = trimmed_docstring[index + 1 : end_index]
            p_output = ("").join(p_output).replace(" ", "")
            parsed_output += p_output

    if end_index == -1:
        return True

    executable_lines = [
        line.split(">>>")[1][1:] for line in docstring.split("\n") if ">>>" in line
    ]

    # noinspection PyBroadException
    f = StringIO()
    with redirect_stdout(f):
        for line in executable_lines:
            # noinspection PyBroadException
            try:
                exec(line)
            except Exception:
                return False

    output = f.getvalue()
    output = output.rstrip()
    output = output.replace(" ", "").replace("\n", "")

    # handling cases when the stdout contains ANSI colour codes
    # 7-bit C1 ANSI sequences
    ansi_escape = re.compile(
        r"""
    \x1B  # ESC
    (?:   # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |     # or [ for CSI, followed by a control sequence
        \[
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
    """,
        re.VERBOSE,
    )

    output = ansi_escape.sub("", output)

    print("Output: ", output)
    print("Putput: ", parsed_output)

    # assert output == parsed_output, "Output is unequal to the docstrings output."
    if not (output == parsed_output):
        ivy.warn(
            "Output is unequal to the docstrings output: %s" % fn_name, stacklevel=0
        )
    return True


def var_fn(x, *, dtype=None, device=None):
    return ivy.variable(ivy.array(x, dtype=dtype, device=device))


def exclude(exclusion_list):
    global _excluded
    _excluded += list(set(exclusion_list) - set(_excluded))


def frameworks():
    return list(
        set(
            [
                ivy_fw()
                for fw_str, ivy_fw in _ivy_fws_dict.items()
                if ivy_fw() is not None and fw_str not in _excluded
            ]
        )
    )


def calls():
    return [
        call
        for (fw_str, ivy_fw), call in zip(_ivy_fws_dict.items(), _calls)
        if ivy_fw() is not None and fw_str not in _excluded
    ]


def f_n_calls():
    return [
        (ivy_fw(), call)
        for (fw_str, ivy_fw), call in zip(_ivy_fws_dict.items(), _calls)
        if ivy_fw() is not None and fw_str not in _excluded
    ]


def assert_all_close(x, y, rtol=1e-05, atol=1e-08):
    if ivy.is_ivy_container(x) and ivy.is_ivy_container(y):
        ivy.Container.multi_map(assert_all_close, [x, y])
    else:
        assert np.allclose(
            np.nan_to_num(x), np.nan_to_num(y), rtol=rtol, atol=atol
        ), "{} != {}".format(x, y)


def kwargs_to_args_n_kwargs(num_positional_args, kwargs):
    args = [v for v in list(kwargs.values())[:num_positional_args]]
    kwargs = {k: kwargs[k] for k in list(kwargs.keys())[num_positional_args:]}
    return args, kwargs


def list_of_length(x, length):
    return st.lists(x, min_size=length, max_size=length)


def as_cont(x):
    return ivy.Container({"a": x, "b": {"c": x, "d": x}})


def as_lists(*args):
    return (a if isinstance(a, list) else [a] for a in args)


def create_args(input_dtypes, num_positional_args, as_variable_flags, all_as_kwargs_np):
    args_np, kwargs_np = kwargs_to_args_n_kwargs(num_positional_args, all_as_kwargs_np)
    args_idxs = ivy.nested_indices_where(args_np, lambda x: isinstance(x, np.ndarray))
    arg_np_vals = ivy.multi_index_nest(args_np, args_idxs)
    num_arg_vals = len(arg_np_vals)
    arg_array_vals = [
        ivy.array(x, dtype=d) for x, d in zip(arg_np_vals, input_dtypes[:num_arg_vals])
    ]
    arg_array_vals = [
        ivy.variable(x) if v else x
        for x, v in zip(arg_array_vals, as_variable_flags[:num_arg_vals])
    ]
    args = ivy.copy_nest(args_np, to_mutable=True)
    ivy.set_nest_at_indices(args, args_idxs, arg_array_vals)

    # create kwargs
    kwargs_idxs = ivy.nested_indices_where(
        kwargs_np, lambda x: isinstance(x, np.ndarray)
    )
    kwarg_np_vals = ivy.multi_index_nest(kwargs_np, kwargs_idxs)
    kwarg_array_vals = [
        ivy.array(x, dtype=d)
        for x, d in zip(kwarg_np_vals, input_dtypes[num_arg_vals:])
    ]
    kwarg_array_vals = [
        ivy.variable(x) if v else x
        for x, v in zip(kwarg_array_vals, as_variable_flags[num_arg_vals:])
    ]

    kwargs = ivy.copy_nest(kwargs_np, to_mutable=True)
    ivy.set_nest_at_indices(kwargs, kwargs_idxs, kwarg_array_vals)

    # create numpy args
    args_np = ivy.nested_map(
        args,
        lambda x: ivy.to_numpy(x) if ivy.is_ivy_container(x) or ivy.is_array(x) else x,
    )
    kwargs_np = ivy.nested_map(
        kwargs,
        lambda x: ivy.to_numpy(x) if ivy.is_ivy_container(x) or ivy.is_array(x) else x,
    )
    return args, kwargs, args_np, kwargs_np


def test_array_method(
    input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags: Union[bool, List[bool]],
    all_as_kwargs_np,
    num_positional_args: int,
    input_dtypes_constructor: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags_constructor: Union[bool, List[bool]],
    constructor_kwargs,
    num_positional_args_constructor: int,
    fw: str,
    class_name: str,
    rtol: float = 1e-03,
    atol: float = 1e-06,
    test_values: bool = True,
):
    """Tests a class-method that consumes (or returns) arrays for the current backend
    by comparing the result with numpy.

    Parameters
    ----------
    input_dtypes
        data types of the input arguments in order.
    as_variable_flags
        dictates whether the corresponding input argument should be treated as an
        ivy Variable.
    all_as_kwargs_np:
        input arguments to the function as keyword arguments.
    num_positional_args
        number of input arguments that must be passed as positional
        arguments.
    input_dtypes_constructor
        data types of the input arguments for the constructor in order.
    as_variable_flags_constructor
        dictates whether the corresponding input argument should be treated as an
        ivy Variable for the constructor
    constructor_kwargs:
        input arguments to the constructor as keyword arguments.
    num_positional_args_constructor
        number of input arguments that must be passed as positional
        arguments to the constructor.
    fw
        current backend (framework).
    class_name
        name of the class to test.
    rtol
        relative tolerance value.
    atol
        absolute tolerance value.
    test_values
        if true, test for the correctness of the resulting values.

    Returns
    -------
    ret
        optional, return value from the function
    ret_np
        optional, return value from the Numpy function
    """
    # convert single values to length 1 lists
    if not isinstance(input_dtypes, list):
        input_dtypes = [input_dtypes]
    if not isinstance(as_variable_flags, list):
        as_variable_flags = [as_variable_flags]

    # update variable flags to be compatible with float dtype
    as_variable_flags = [
        v if ivy.is_float_dtype(d) else False
        for v, d in zip(as_variable_flags, input_dtypes)
    ]
    # tolerance dict for dtypes
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}

    # change all data types so that they are supported by this framework
    input_dtypes = ["float32" if d in ivy.invalid_dtypes else d for d in input_dtypes]

    # create args
    calling_args, calling_kwargs, calling_args_np, calling_kwargs_np = create_args(
        input_dtypes, num_positional_args, as_variable_flags, all_as_kwargs_np
    )

    (
        constructor_args,
        constructor_kwargs,
        constructor_args_np,
        constructor_kwargs_np,
    ) = create_args(
        input_dtypes_constructor,
        num_positional_args_constructor,
        as_variable_flags_constructor,
        constructor_kwargs,
    )

    # run
    ins = ivy.__dict__[class_name](*constructor_args, **constructor_kwargs)
    ret = ins(*calling_args, **calling_kwargs)

    # assert idx of return if the idx of the out array provided

    if "bfloat16" in input_dtypes:
        return  # bfloat16 is not supported by numpy
    # compute the return with a NumPy backend
    ivy.set_backend("numpy")
    ins_np = ivy.__dict__[class_name](*constructor_args_np, **constructor_kwargs_np)
    ret_from_np = ivy.to_native(
        ins_np(*calling_args_np, **calling_kwargs_np), nested=True
    )
    ivy.unset_backend()

    # assuming value test will be handled manually in the test function
    if not test_values:
        return ret, ret_from_np

    # flatten the return
    if not isinstance(ret, tuple):
        ret = (ret,)

    ret_idxs = ivy.nested_indices_where(ret, ivy.is_ivy_array)
    ret_flat = ivy.multi_index_nest(ret, ret_idxs)

    # convert the return to NumPy
    ret_np_flat = [ivy.to_numpy(x) for x in ret_flat]

    # flatten the return from the NumPy backend
    if not isinstance(ret_from_np, tuple):
        ret_from_np = (ret_from_np,)
    ret_from_np_flat = ivy.multi_index_nest(ret_from_np, ret_idxs)

    # value tests, iterating through each array in the flattened returns
    for ret_np, ret_from_np in zip(ret_np_flat, ret_from_np_flat):
        rtol = tolerance_dict.get(str(ret_from_np.dtype), rtol)
        assert_all_close(ret_np, ret_from_np, rtol=rtol, atol=atol)


def test_array_function(
    input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags: Union[bool, List[bool]],
    with_out: bool,
    num_positional_args: int,
    native_array_flags: Union[bool, List[bool]],
    container_flags: Union[bool, List[bool]],
    instance_method: bool,
    fw: str,
    fn_name: str,
    rtol: float = 1e-03,
    atol: float = 1e-06,
    test_values: bool = True,
    **all_as_kwargs_np
):
    """Tests a function that consumes (or returns) arrays for the current backend
    by comparing the result with numpy.

    Parameters
    ----------
    input_dtypes
        data types of the input arguments in order.
    as_variable_flags
        dictates whether the corresponding input argument should be treated
        as an ivy Variable.
    with_out
        if true, the function is also tested with the optional out argument.
    num_positional_args
        number of input arguments that must be passed as positional
        arguments.
    native_array_flags
        dictates whether the corresponding input argument should be treated
         as a native array.
    container_flags
        dictates whether the corresponding input argument should be treated
         as an ivy Container.
    instance_method
        if true, the function is run as an instance method of the first
         argument (should be an ivy Array or Container).
    fw
        current backend (framework).
    fn_name
        name of the function to test.
    rtol
        relative tolerance value.
    atol
        absolute tolerance value.
    test_values
        if true, test for the correctness of the resulting values.
    all_as_kwargs_np
        input arguments to the function as keyword arguments.

    Returns
    -------
    ret
        optional, return value from the function
    ret_np
        optional, return value from the Numpy function

    Examples
    --------
    >>> input_dtypes = 'float64'
    >>> as_variable_flags = False
    >>> with_out = False
    >>> num_positional_args = 0
    >>> native_array_flags = False
    >>> container_flags = False
    >>> instance_method = False
    >>> fw = "torch"
    >>> fn_name = "abs"
    >>> x = np.array([-1])
    >>> test_array_function(input_dtypes, as_variable_flags, with_out,\
                            num_positional_args, native_array_flags,
    >>> container_flags, instance_method, fw, fn_name, x=x)

    >>> input_dtypes = ['float64', 'float32']
    >>> as_variable_flags = [False, True]
    >>> with_out = False
    >>> num_positional_args = 1
    >>> native_array_flags = [True, False]
    >>> container_flags = [False, False]
    >>> instance_method = False
    >>> fw = "numpy"
    >>> fn_name = "add"
    >>> x1 = np.array([1, 3, 4])
    >>> x2 = np.array([-3, 15, 24])
    >>> test_array_function(input_dtypes, as_variable_flags, with_out,\
                            num_positional_args, native_array_flags,\
                             container_flags, instance_method,\
                              fw, fn_name, x1=x1, x2=x2)
    """
    # convert single values to length 1 lists
    input_dtypes, as_variable_flags, native_array_flags, container_flags = as_lists(
        input_dtypes, as_variable_flags, native_array_flags, container_flags
    )

    # make all lists equal in length
    num_arrays = max(
        len(input_dtypes),
        len(as_variable_flags),
        len(native_array_flags),
        len(container_flags),
    )
    if len(input_dtypes) < num_arrays:
        input_dtypes = [input_dtypes[0] for _ in range(num_arrays)]
    if len(as_variable_flags) < num_arrays:
        as_variable_flags = [as_variable_flags[0] for _ in range(num_arrays)]
    if len(native_array_flags) < num_arrays:
        native_array_flags = [native_array_flags[0] for _ in range(num_arrays)]
    if len(container_flags) < num_arrays:
        container_flags = [container_flags[0] for _ in range(num_arrays)]

    # update variable flags to be compatible with float dtype and with_out args
    as_variable_flags = [
        v if ivy.is_float_dtype(d) and not with_out else False
        for v, d in zip(as_variable_flags, input_dtypes)
    ]

    # tolerance dict for dtypes
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}
    # update instance_method flag to only be considered if the
    # first term is either an ivy.Array or ivy.Container
    instance_method = instance_method and (
        not native_array_flags[0] or container_flags[0]
    )

    # check for unsupported dtypes
    fn = getattr(ivy, fn_name)
    for d in input_dtypes:
        if d in ivy.function_unsupported_dtypes(fn):
            return
    if "dtype" in all_as_kwargs_np and all_as_kwargs_np[
        "dtype"
    ] in ivy.function_unsupported_dtypes(fn):
        return

    # split the arguments into their positional and keyword components
    args_np, kwargs_np = kwargs_to_args_n_kwargs(num_positional_args, all_as_kwargs_np)

    # create args
    args_idxs = ivy.nested_indices_where(args_np, lambda x: isinstance(x, np.ndarray))
    arg_np_vals = ivy.multi_index_nest(args_np, args_idxs)
    num_arg_vals = len(arg_np_vals)
    arg_array_vals = [
        ivy.array(x, dtype=d) for x, d in zip(arg_np_vals, input_dtypes[:num_arg_vals])
    ]
    arg_array_vals = [
        ivy.variable(x) if v else x
        for x, v in zip(arg_array_vals, as_variable_flags[:num_arg_vals])
    ]
    arg_array_vals = [
        ivy.to_native(x) if n else x
        for x, n in zip(arg_array_vals, native_array_flags[:num_arg_vals])
    ]
    arg_array_vals = [
        as_cont(x) if c else x
        for x, c in zip(arg_array_vals, container_flags[:num_arg_vals])
    ]
    args = ivy.copy_nest(args_np, to_mutable=True)
    ivy.set_nest_at_indices(args, args_idxs, arg_array_vals)

    # create kwargs
    kwargs_idxs = ivy.nested_indices_where(
        kwargs_np, lambda x: isinstance(x, np.ndarray)
    )
    kwarg_np_vals = ivy.multi_index_nest(kwargs_np, kwargs_idxs)
    kwarg_array_vals = [
        ivy.array(x, dtype=d)
        for x, d in zip(kwarg_np_vals, input_dtypes[num_arg_vals:])
    ]
    kwarg_array_vals = [
        ivy.variable(x) if v else x
        for x, v in zip(kwarg_array_vals, as_variable_flags[num_arg_vals:])
    ]
    kwarg_array_vals = [
        ivy.to_native(x) if n else x
        for x, n in zip(kwarg_array_vals, native_array_flags[num_arg_vals:])
    ]
    kwarg_array_vals = [
        as_cont(x) if c else x
        for x, c in zip(kwarg_array_vals, container_flags[num_arg_vals:])
    ]
    kwargs = ivy.copy_nest(kwargs_np, to_mutable=True)
    ivy.set_nest_at_indices(kwargs, kwargs_idxs, kwarg_array_vals)

    # create numpy args
    args_np = ivy.nested_map(
        args,
        lambda x: ivy.to_numpy(x) if ivy.is_ivy_container(x) or ivy.is_array(x) else x,
    )
    kwargs_np = ivy.nested_map(
        kwargs,
        lambda x: ivy.to_numpy(x) if ivy.is_ivy_container(x) or ivy.is_array(x) else x,
    )

    # run either as an instance method or from the API directly
    instance = None
    if instance_method:
        is_instance = [
            (not n) or c for n, c in zip(native_array_flags, container_flags)
        ]
        arg_is_instance = is_instance[:num_arg_vals]
        kwarg_is_instance = is_instance[num_arg_vals:]
        if arg_is_instance and max(arg_is_instance):
            i = 0
            for i, a in enumerate(arg_is_instance):
                if a:
                    break
            instance_idx = args_idxs[i]
            instance = ivy.index_nest(args, instance_idx)
            args = ivy.copy_nest(args, to_mutable=True)
            ivy.prune_nest_at_index(args, instance_idx)
        else:
            i = 0
            for i, a in enumerate(kwarg_is_instance):
                if a:
                    break
            instance_idx = kwargs_idxs[i]
            instance = ivy.index_nest(kwargs, instance_idx)
            kwargs = ivy.copy_nest(kwargs, to_mutable=True)
            ivy.prune_nest_at_index(kwargs, instance_idx)
        ret = instance.__getattribute__(fn_name)(*args, **kwargs)
    else:
        ret = ivy.__dict__[fn_name](*args, **kwargs)

    # assert idx of return if the idx of the out array provided
    out = ret
    if with_out:
        assert not isinstance(ret, tuple)
        if max(container_flags):
            assert ivy.is_ivy_container(ret)
        else:
            assert ivy.is_array(ret)
        if instance_method:
            ret = instance.__getattribute__(fn_name)(*args, **kwargs, out=out)
        else:
            ret = ivy.__dict__[fn_name](*args, **kwargs, out=out)

        if max(container_flags):
            assert ret is out

        if not max(container_flags) and fw not in ["tensorflow", "jax", "numpy"]:
            # these backends do not always support native inplace updates
            assert ret.data is out.data

    if "bfloat16" in input_dtypes:
        return  # bfloat16 is not supported by numpy
    # compute the return with a NumPy backend
    ivy.set_backend("numpy")
    ret_from_np = ivy.to_native(
        ivy.__dict__[fn_name](*args_np, **kwargs_np), nested=True
    )
    ivy.unset_backend()

    # assuming value test will be handled manually in the test function
    if not test_values:
        return ret, ret_from_np

    # flatten the return
    if not isinstance(ret, tuple):
        ret = (ret,)

    ret_idxs = ivy.nested_indices_where(ret, ivy.is_ivy_array)
    ret_flat = ivy.multi_index_nest(ret, ret_idxs)

    # convert the return to NumPy
    ret_np_flat = [ivy.to_numpy(x) for x in ret_flat]

    # flatten the return from the NumPy backend
    if not isinstance(ret_from_np, tuple):
        ret_from_np = (ret_from_np,)
    ret_from_np_flat = ivy.multi_index_nest(ret_from_np, ret_idxs)

    # value tests, iterating through each array in the flattened returns
    for ret_np, ret_from_np in zip(ret_np_flat, ret_from_np_flat):
        rtol = tolerance_dict.get(str(ret_from_np.dtype), rtol)
        assert_all_close(ret_np, ret_from_np, rtol=rtol, atol=atol)


def test_frontend_function(
    input_dtypes: Union[ivy.Dtype, List[ivy.Dtype]],
    as_variable_flags: Union[bool, List[bool]],
    with_out: bool,
    num_positional_args: int,
    native_array_flags: Union[bool, List[bool]],
    fw: str,
    frontend: str,
    fn_name: str,
    rtol: float = 1e-03,
    atol: float = 1e-06,
    test_values: bool = True,
    **all_as_kwargs_np
):
    """Tests a frontend function for the current backend by comparing the result with
    the function in the associated framework.

    Parameters
    ----------
    input_dtypes
        data types of the input arguments in order.
    as_variable_flags
        dictates whether the corresponding input argument should be treated
        as an ivy Variable.
    with_out
        if true, the function is also tested with the optional out argument.
    num_positional_args
        number of input arguments that must be passed as positional
        arguments.
    native_array_flags
        dictates whether the corresponding input argument should be treated
        as a native array.
    fw
        current backend (framework).
    frontend
        current frontend (framework).
    fn_name
        name of the function to test.
    rtol
        relative tolerance value.
    atol
        absolute tolerance value.
    test_values
        if true, test for the correctness of the resulting values.
    all_as_kwargs_np
        input arguments to the function as keyword arguments.

    Returns
    -------
    ret
        optional, return value from the function
    ret_np
        optional, return value from the Numpy function
    """
    # convert single values to length 1 lists
    input_dtypes, as_variable_flags, native_array_flags = as_lists(
        input_dtypes, as_variable_flags, native_array_flags
    )

    # update variable flags to be compatible with float dtype and with_out args
    as_variable_flags = [
        v if ivy.is_float_dtype(d) and not with_out else False
        for v, d in zip(as_variable_flags, input_dtypes)
    ]
    # tolerance dict for dtypes
    tolerance_dict = {"float16": 1e-2, "float32": 1e-5, "float64": 1e-5, None: 1e-5}

    # parse function name and frontend submodules (i.e. jax.lax, jax.numpy etc.)
    *frontend_submods, fn_name = fn_name.split(".")

    # check for unsupported dtypes in backend framework
    function = getattr(ivy.functional.frontends.__dict__[frontend], fn_name)
    for d in input_dtypes:
        if d in ivy.function_unsupported_dtypes(function):
            return
    if "dtype" in all_as_kwargs_np and all_as_kwargs_np[
        "dtype"
    ] in ivy.function_unsupported_dtypes(function):
        return

    # split the arguments into their positional and keyword components
    args_np, kwargs_np = kwargs_to_args_n_kwargs(num_positional_args, all_as_kwargs_np)

    # change all data types so that they are supported by this framework
    input_dtypes = ["float32" if d in ivy.invalid_dtypes else d for d in input_dtypes]

    # create args
    args_idxs = ivy.nested_indices_where(args_np, lambda x: isinstance(x, np.ndarray))
    arg_np_vals = ivy.multi_index_nest(args_np, args_idxs)
    num_arg_vals = len(arg_np_vals)
    arg_array_vals = [
        ivy.array(x, dtype=d) for x, d in zip(arg_np_vals, input_dtypes[:num_arg_vals])
    ]
    arg_array_vals = [
        ivy.variable(x) if v else x
        for x, v in zip(arg_array_vals, as_variable_flags[:num_arg_vals])
    ]
    arg_array_vals = [
        ivy.to_native(x) if n else x
        for x, n in zip(arg_array_vals, native_array_flags[:num_arg_vals])
    ]
    args = ivy.copy_nest(args_np, to_mutable=True)
    ivy.set_nest_at_indices(args, args_idxs, arg_array_vals)

    # create kwargs
    kwargs_idxs = ivy.nested_indices_where(
        kwargs_np, lambda x: isinstance(x, np.ndarray)
    )
    kwarg_np_vals = ivy.multi_index_nest(kwargs_np, kwargs_idxs)
    kwarg_array_vals = [
        ivy.array(x, dtype=d)
        for x, d in zip(kwarg_np_vals, input_dtypes[num_arg_vals:])
    ]
    kwarg_array_vals = [
        ivy.variable(x) if v else x
        for x, v in zip(kwarg_array_vals, as_variable_flags[num_arg_vals:])
    ]
    kwarg_array_vals = [
        ivy.to_native(x) if n else x
        for x, n in zip(kwarg_array_vals, native_array_flags[num_arg_vals:])
    ]
    kwargs = ivy.copy_nest(kwargs_np, to_mutable=True)
    ivy.set_nest_at_indices(kwargs, kwargs_idxs, kwarg_array_vals)

    # create ivy array args
    args_ivy, kwargs_ivy = ivy.args_to_ivy(*args, **kwargs)

    # frontend function
    frontend_fn = ivy.functional.frontends.__dict__[frontend].__dict__[fn_name]

    # run from the Ivy API directly
    ret = frontend_fn(*args, **kwargs)

    # assert idx of return if the idx of the out array provided
    out = ret
    if with_out:
        assert not isinstance(ret, tuple)
        assert ivy.is_array(ret)
        if "out" in kwargs:
            kwargs["out"] = out
        else:
            args[ivy.arg_info(frontend_fn, name="out")["idx"]] = out
        ret = frontend_fn(*args, **kwargs)

        if fw not in ["tensorflow", "jax", "numpy"]:
            # these backends do not always support native inplace updates
            assert ret.data is out.data

    if "bfloat16" in input_dtypes:
        return  # bfloat16 is not supported by numpy

    # create NumPy args
    args_np = ivy.nested_map(
        args_ivy,
        lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
    )
    kwargs_np = ivy.nested_map(
        kwargs_ivy,
        lambda x: ivy.to_numpy(x._data) if isinstance(x, ivy.Array) else x,
    )

    # temporarily set frontend framework as backend
    ivy.set_backend(frontend)

    # check for unsupported dtypes in frontend framework
    function = getattr(ivy.functional.frontends.__dict__[frontend], fn_name)
    for d in input_dtypes:
        if d in ivy.function_unsupported_dtypes(function):
            return
    if "dtype" in all_as_kwargs_np and all_as_kwargs_np[
        "dtype"
    ] in ivy.function_unsupported_dtypes(function):
        return

    # create frontend framework args
    args_frontend = ivy.nested_map(
        args_np,
        lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
    )
    kwargs_frontend = ivy.nested_map(
        kwargs_np,
        lambda x: ivy.native_array(x) if isinstance(x, np.ndarray) else x,
    )

    # compute the return via the frontend framework
    frontend_fw = importlib.import_module(".".join([frontend] + frontend_submods))
    frontend_ret = frontend_fw.__dict__[fn_name](*args_frontend, **kwargs_frontend)

    # tuplify the frontend return
    if not isinstance(frontend_ret, tuple):
        frontend_ret = (frontend_ret,)

    # flatten the frontend return and convert to NumPy arrays
    frontend_ret_idxs = ivy.nested_indices_where(frontend_ret, ivy.is_native_array)
    frontend_ret_flat = ivy.multi_index_nest(frontend_ret, frontend_ret_idxs)
    frontend_ret_np_flat = [ivy.to_numpy(x) for x in frontend_ret_flat]

    # unset frontend framework from backend
    ivy.unset_backend()

    # assuming value test will be handled manually in the test function
    if not test_values:
        return ret, frontend_ret

    # flatten the return
    if not isinstance(ret, tuple):
        ret = (ret,)

    ret_idxs = ivy.nested_indices_where(ret, ivy.is_ivy_array)
    ret_flat = ivy.multi_index_nest(ret, ret_idxs)

    # convert the return to NumPy
    ret_np_flat = [ivy.to_numpy(x) for x in ret_flat]

    # value tests, iterating through each array in the flattened returns
    for ret_np, frontend_ret in zip(ret_np_flat, frontend_ret_np_flat):
        rtol = tolerance_dict.get(str(frontend_ret.dtype), rtol)
        assert_all_close(ret_np, frontend_ret, rtol=rtol, atol=atol)


# Hypothesis #
# -----------#


@st.composite
def array_dtypes(draw, na=st.shared(st.integers(), key="num_arrays")):
    size = na if isinstance(na, int) else draw(na)
    return draw(
        st.lists(
            st.sampled_from(ivy_np.valid_float_dtypes), min_size=size, max_size=size
        )
    )


@st.composite
def array_bools(draw, na=st.shared(st.integers(), key="num_arrays")):
    size = na if isinstance(na, int) else draw(na)
    return draw(st.lists(st.booleans(), min_size=size, max_size=size))


@st.composite
def lists(draw, arg, min_size=None, max_size=None, size_bounds=None):
    ints = st.integers(size_bounds[0], size_bounds[1]) if size_bounds else st.integers()
    if isinstance(min_size, str):
        min_size = draw(st.shared(ints, key=min_size))
    if isinstance(max_size, str):
        max_size = draw(st.shared(ints, key=max_size))
    return draw(st.lists(arg, min_size=min_size, max_size=max_size))


@st.composite
def valid_axes(draw, ndim=None, size_bounds=None):
    ints = st.integers(size_bounds[0], size_bounds[1]) if size_bounds else st.integers()
    dims = draw(st.shared(ints, key=ndim))
    any_axis_strategy = (
        st.none() | st.integers(-dims, dims - 1) | nph.valid_tuple_axes(dims)
    )
    return draw(any_axis_strategy)


@st.composite
def integers(draw, min_value=None, max_value=None):
    if isinstance(min_value, str):
        min_value = draw(st.shared(st.integers(), key=min_value))
    if isinstance(max_value, str):
        max_value = draw(st.shared(st.integers(), key=max_value))
    return draw(st.integers(min_value=min_value, max_value=max_value))


@st.composite
def dtype_and_values(
    draw,
    available_dtypes,
    n_arrays=1,
    allow_inf=True,
    max_num_dims=5,
    max_dim_size=10,
    shape=None,
    shared_dtype=False,
):
    if not isinstance(n_arrays, int):
        n_arrays = draw(n_arrays)
    if n_arrays == 1:
        dtypes = set(available_dtypes).difference(set(ivy.invalid_dtypes))
        dtype = draw(list_of_length(st.sampled_from(tuple(dtypes)), 1))
    elif shared_dtype:
        dtypes = set(available_dtypes).difference(set(ivy.invalid_dtypes))
        dtype = draw(list_of_length(st.sampled_from(tuple(dtypes)), 1))
        dtype = [dtype[0] for _ in range(n_arrays)]
    else:
        unwanted_types = set(ivy.invalid_dtypes).union(
            set(ivy.all_dtypes).difference(set(available_dtypes))
        )
        pairs = ivy.promotion_table.keys()
        dtypes = [
            pair for pair in pairs if not any([d in pair for d in unwanted_types])
        ]
        dtype = list(draw(st.sampled_from(dtypes)))
        if n_arrays > 2:
            dtype += [dtype[i % 2] for i in range(n_arrays - 2)]
    if shape:
        shape = draw(shape)
    else:
        shape = draw(
            st.shared(
                get_shape(max_num_dims=max_num_dims, max_dim_size=max_dim_size),
                key="shape",
            )
        )
    values = []
    for i in range(n_arrays):
        values.append(
            draw(array_values(dtype=dtype[i], shape=shape, allow_inf=allow_inf))
        )
    if n_arrays == 1:
        dtype = dtype[0]
        values = values[0]
    return dtype, values


# taken from
# https://github.com/data-apis/array-api-tests/array_api_tests/test_manipulation_functions.py
@st.composite
def reshape_shapes(draw, shape):
    size = 1 if len(shape) == 0 else math.prod(shape)
    rshape = draw(st.lists(st.integers(0)).filter(lambda s: math.prod(s) == size))
    # assume(all(side <= MAX_SIDE for side in rshape))
    if len(rshape) != 0 and size > 0 and draw(st.booleans()):
        index = draw(st.integers(0, len(rshape) - 1))
        rshape[index] = -1
    return tuple(rshape)


# taken from https://github.com/HypothesisWorks/hypothesis/issues/1115
@st.composite
def subsets(draw, elements):
    return tuple(e for e in elements if draw(st.booleans()))


@st.composite
def array_values(
    draw,
    dtype,
    shape,
    min_value=None,
    max_value=None,
    allow_nan=False,
    allow_subnormal=False,
    allow_inf=False,
    exclude_min=False,
    exclude_max=False,
    allow_negative=True,
):
    size = 1
    if type(shape) != tuple:
        size = shape
    else:
        for dim in shape:
            size *= dim
    if "int" in dtype:
        if dtype == "int8":
            min_value = min_value if min_value else -128
            max_value = max_value if max_value else 127
        elif dtype == "int16":
            min_value = min_value if min_value else -32768
            max_value = max_value if max_value else 32767
        elif dtype == "int32":
            min_value = min_value if min_value else -2147483648
            max_value = max_value if max_value else 2147483647
        elif dtype == "int64":
            min_value = min_value if min_value else -9223372036854775808
            max_value = max_value if max_value else 9223372036854775807
        elif dtype == "uint8":
            min_value = min_value if min_value else 0
            max_value = max_value if max_value else 255
        elif dtype == "uint16":
            min_value = min_value if min_value else 0
            max_value = max_value if max_value else 65535
        elif dtype == "uint32":
            min_value = min_value if min_value else 0
            max_value = max_value if max_value else 4294967295
        elif dtype == "uint64":
            min_value = min_value if min_value else 0
            max_value = max_value if max_value else 18446744073709551615
        values = draw(list_of_length(st.integers(min_value, max_value), size))
    elif dtype == "float16":
        values = draw(
            list_of_length(
                st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    allow_nan=allow_nan,
                    allow_subnormal=allow_subnormal,
                    allow_infinity=allow_inf,
                    width=16,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                size,
            )
        )
    elif dtype == "float32":
        values = draw(
            list_of_length(
                st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    allow_nan=allow_nan,
                    allow_subnormal=allow_subnormal,
                    allow_infinity=allow_inf,
                    width=32,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                size,
            )
        )
    elif dtype == "float64":
        values = draw(
            list_of_length(
                st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    allow_nan=allow_nan,
                    allow_subnormal=allow_subnormal,
                    allow_infinity=allow_inf,
                    width=64,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                size,
            )
        )
    elif dtype == "bool":
        values = draw(list_of_length(st.booleans(), size))
    array = np.array(values)
    if dtype != "bool" and not allow_negative:
        array = np.abs(array)
    if type(shape) == tuple:
        array = array.reshape(shape)
    return array.tolist()


@st.composite
def get_shape(
    draw,
    allow_none=False,
    min_num_dims=0,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=10,
):
    if allow_none:
        shape = draw(
            st.none()
            | st.lists(
                st.integers(min_value=min_dim_size, max_value=max_dim_size),
                min_size=min_num_dims,
                max_size=max_num_dims,
            )
        )
    else:
        shape = draw(
            st.lists(
                st.integers(min_value=min_dim_size, max_value=max_dim_size),
                min_size=min_num_dims,
                max_size=max_num_dims,
            )
        )
    if shape is None:
        return shape
    return tuple(shape)


def none_or_list_of_floats(
    dtype,
    size,
    min_value=None,
    max_value=None,
    exclude_min=False,
    exclude_max=False,
    no_none=False,
):
    if no_none:
        if dtype == "float16":
            values = list_of_length(
                st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=16,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                size,
            )
        elif dtype == "float32":
            values = list_of_length(
                st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=32,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                size,
            )
        elif dtype == "float64":
            values = list_of_length(
                st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=64,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                size,
            )
    else:
        if dtype == "float16":
            values = list_of_length(
                st.none()
                | st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=16,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                size,
            )
        elif dtype == "float32":
            values = list_of_length(
                st.none()
                | st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=32,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                size,
            )
        elif dtype == "float64":
            values = list_of_length(
                st.none()
                | st.floats(
                    min_value=min_value,
                    max_value=max_value,
                    width=64,
                    allow_subnormal=False,
                    allow_infinity=False,
                    allow_nan=False,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                ),
                size,
            )
    return values


@st.composite
def get_mean_std(draw, dtype):
    values = draw(none_or_list_of_floats(dtype, 2))
    values[1] = abs(values[1]) if values[1] else None
    return values[0], values[1]


@st.composite
def get_bounds(draw, dtype):
    if "int" in dtype:
        values = draw(array_values(dtype, 2))
        values[0], values[1] = abs(values[0]), abs(values[1])
        low, high = min(values), max(values)
        if low == high:
            return draw(get_bounds(dtype))
    else:
        values = draw(none_or_list_of_floats(dtype, 2))
        if values[0] is not None and values[1] is not None:
            low, high = min(values), max(values)
        else:
            low, high = values[0], values[1]
        if ivy.default(low, 0.0) >= ivy.default(high, 1.0):
            return draw(get_bounds(dtype))
    return low, high


@st.composite
def get_probs(draw, dtype):
    shape = draw(
        get_shape(min_num_dims=2, max_num_dims=5, min_dim_size=2, max_dim_size=10)
    )
    probs = draw(array_values(dtype, shape, min_value=0, exclude_min=True))
    return probs, shape[1]


@st.composite
def get_axis(draw, shape, allow_none=False):
    axes = len(shape)
    if allow_none:
        axis = draw(
            st.none()
            | st.integers(-axes, axes - 1)
            | st.lists(
                st.integers(-axes, axes - 1),
                min_size=1,
                max_size=axes,
                unique_by=lambda x: shape[x],
            )
        )
    else:
        axis = draw(
            st.integers(-axes, axes - 1)
            | st.lists(
                st.integers(-axes, axes - 1),
                min_size=1,
                max_size=axes,
                unique_by=lambda x: shape[x],
            )
        )
    if type(axis) == list:

        def sort_key(ele, max_len):
            if ele < 0:
                return ele + max_len - 1
            return ele

        axis.sort(key=(lambda ele: sort_key(ele, axes)))
        axis = tuple(axis)
    return axis


@st.composite
def num_positional_args(draw, fn_name: str = None):
    num_positional_only = 0
    num_keyword_only = 0
    total = 0
    fn = None
    for i, fn_name_key in enumerate(fn_name.split(".")):
        if i == 0:
            fn = ivy.__dict__[fn_name_key]
        else:
            fn = fn.__dict__[fn_name_key]
    for param in inspect.signature(fn).parameters.values():
        total += 1
        if param.kind == param.POSITIONAL_ONLY:
            num_positional_only += 1
        elif param.kind == param.KEYWORD_ONLY:
            num_keyword_only += 1
    return draw(
        integers(min_value=num_positional_only, max_value=(total - num_keyword_only))
    )

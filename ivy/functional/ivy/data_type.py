# global
import math
import numpy as np
from numbers import Number
from typing import Union, Tuple, List, Optional, Callable

# local
import ivy
from ivy.backend_handler import current_backend
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    inputs_to_native_arrays,
    handle_nestable,
)

# Array API Standard #
# -------------------#

Finfo = None
Iinfo = None


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def astype(
    x: Union[ivy.Array, ivy.NativeArray],
    dtype: Union[ivy.Dtype, ivy.NativeDtype],
    *,
    copy: bool = True,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Copies an array to a specified data type irrespective of :ref:`type-promotion`
    rules.

    .. note::
       Casting floating-point ``NaN`` and ``infinity`` values to integral data types is
       not specified and is implementation-dependent.

    .. note::
       When casting a boolean input array to a numeric data type, a value of ``True``
       must cast to a numeric value equal to ``1``, and a value of ``False`` must cast
       to a numeric value equal to ``0``.

       When casting a numeric input array to ``bool``, a value of ``0`` must cast to
       ``False``, and a non-zero value must cast to ``True``.

    Parameters
    ----------
    x
        array to cast.
    dtype
        desired data type.
    copy
        specifies whether to copy an array when the specified ``dtype`` matches the data
        type of the input array ``x``. If ``True``, a newly allocated array must always
        be returned. If ``False`` and the specified ``dtype`` matches the data type of
        the input array, the input array must be returned; otherwise, a newly allocated
        must be returned. Default: ``True``.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having the specified data type. The returned array must have the same
        shape as ``x``.

    Examples
    --------
    >>> x = ivy.array([1, 2])
    >>> dtype = ivy.float64
    >>> y = ivy.astype(x, dtype = dtype)
    >>> print(y)
    ivy.array([1., 2.])
    """
    return current_backend(x).astype(x, dtype, copy=copy, out=out)


@to_native_arrays_and_back
@handle_nestable
def broadcast_arrays(*arrays: Union[ivy.Array, ivy.NativeArray]) -> List[ivy.Array]:
    """Broadcasts one or more arrays against one another.

    Parameters
    ----------
    arrays
        an arbitrary number of to-be broadcasted arrays.

    Returns
    -------
    ret
        Each array must have the same shape. Each array must have the same dtype as its
        corresponding input array.

    """
    return current_backend(arrays[0]).broadcast_arrays(*arrays)


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
def broadcast_to(
    x: Union[ivy.Array, ivy.NativeArray],
    shape: Tuple[int, ...],
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    """Broadcasts an array to a specified shape.

    Parameters
    ----------
    x
        array to broadcast.
    shape
        array shape. Must be compatible with x (see Broadcasting). If
        the array is incompatible with the specified shape, the function should raise an
        exception.
    out
        optional output array, for writing the result to. It must have a shape that the
        inputs broadcast to.

    Returns
    -------
    ret
        an array having a specified shape. Must have the same data type as x.

    """
    return current_backend(x).broadcast_to(x, shape, out=out)


@inputs_to_native_arrays
@handle_nestable
def can_cast(
    from_: Union[ivy.Dtype, ivy.Array, ivy.NativeArray], to: ivy.Dtype
) -> bool:
    """
    Determines if one data type can be cast to another data type according to
    :ref:`type-promotion` rules.

    Parameters
    ----------
    from_
        input data type or array from which to cast.
    to
        desired data type.

    Returns
    -------
    ret
        ``True`` if the cast can occur according to :ref:`type-promotion` rules;
        otherwise, ``False``.

    This function conforms to the `Array API Standard
    <https://data-apis.org/array-api/latest/>`_. This docstring is an extension of the
    `docstring <https://data-apis.org/array-api/latest/API_specification/generated/signatures.data_type_functions.can_cast.html>`_ # noqa
    in the standard.

    Both the description and the type hints above assumes an array input for simplicity,
    but this function is *nestable*, and therefore also accepts :code:`ivy.Container`
    instances in place of any of the arguments.

    Examples
    --------
     With :code:`ivy.Dtype` input:

    >>> print(ivy.can_cast(ivy.uint8, ivy.int32))
    True

    >>> print(ivy.can_cast(ivy.float64, 'int64'))
    False

    With :code:`ivy.Array` input:

    >>> x = ivy.array([1., 2., 3.])
    >>> print(ivy.can_cast(x, ivy.float64))
    True

    With :code:`ivy.NativeArray` input:

    >>> x = ivy.native_array([[-1, -1, -1], [1, 1, 1]], \
        dtype='int16')
    >>> print(ivy.can_cast(x, 'uint8'))
    False

    With :code:`ivy.Container` input:

    >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
        b=ivy.array([3, 4, 5]))
    >>> print(ivy.can_cast(x, 'int64'))
    {
        a: false,
        b: true
    }
    """
    return current_backend(from_).can_cast(from_, to)


@inputs_to_native_arrays
@handle_nestable
def finfo(type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray]) -> Finfo:
    """Machine limits for floating-point data types.

    Parameters
    ----------
    type
        the kind of floating-point data-type about which to get information.

    Returns
    -------
    ret
        an object having the followng attributes:
        - **bits**: *int*
          number of bits occupied by the floating-point data type.
        - **eps**: *float*
          difference between 1.0 and the next smallest representable floating-point
          number larger than 1.0 according to the IEEE-754 standard.
        - **max**: *float*
          largest representable number.
        - **min**: *float*
          smallest representable number.
        - **smallest_normal**: *float*
          smallest positive floating-point number with full precision.

    """
    return current_backend(None).finfo(type)


@inputs_to_native_arrays
@handle_nestable
def iinfo(type: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray]) -> Iinfo:
    """Machine limits for integer data types.

    Parameters
    ----------
    type
        the kind of integer data-type about which to get information.

    Returns
    -------
    ret
        a class with that encapsules the following attributes:
        - **bits**: *int*
          number of bits occupied by the type.
        - **max**: *int*
          largest representable number.
        - **min**: *int*
          smallest representable number.

    """
    return current_backend(None).iinfo(type)


@inputs_to_native_arrays
@handle_nestable
def result_type(
    *arrays_and_dtypes: Union[ivy.Array, ivy.NativeArray, ivy.Dtype]
) -> ivy.Dtype:
    """Returns the dtype that results from applying the type promotion rules (see
    :ref:`type-promotion`) to the arguments.

    .. note::
       If provided mixed dtypes (e.g., integer and floating-point), the returned dtype
       will be implementation-specific.

    Parameters
    ----------
    arrays_and_dtypes
        an arbitrary number of input arrays and/or dtypes.

    Returns
    -------
    ret
        the dtype resulting from an operation involving the input arrays and dtypes.

    """
    return current_backend(arrays_and_dtypes[0]).result_type(arrays_and_dtypes)


# Extra #
# ------#

default_dtype_stack = list()
default_float_dtype_stack = list()
default_int_dtype_stack = list()


class DefaultDtype:
    """"""

    # noinspection PyShadowingNames
    def __init__(self, dtype: ivy.Dtype):
        self._dtype = dtype

    def __enter__(self):
        set_default_dtype(self._dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_dtype()
        return self


class DefaultFloatDtype:
    """"""

    # noinspection PyShadowingNames
    def __init__(self, float_dtype: ivy.Dtype):
        self._float_dtype = float_dtype

    def __enter__(self):
        set_default_float_dtype(self._float_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_float_dtype()
        return self


class DefaultIntDtype:
    """"""

    # noinspection PyShadowingNames
    def __init__(self, float_dtype: ivy.Dtype):
        self._float_dtype = float_dtype

    def __enter__(self):
        set_default_int_dtype(self._float_dtype)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        unset_default_int_dtype()
        return self


def dtype_bits(dtype_in: Union[ivy.Dtype, str]) -> int:
    """Get the number of bits used for representing the input data type.

    Parameters
    ----------
    dtype_in
        The data type to determine the number of bits for.

    Returns
    -------
    ret
        The number of bits used to represent the data type.

    """
    return current_backend(dtype_in).dtype_bits(dtype_in)


def as_ivy_dtype(dtype_in: Union[ivy.Dtype, str]) -> ivy.Dtype:
    """Convert native data type to string representation.

    Parameters
    ----------
    dtype_in
        The data type to convert to string.

    Returns
    -------
    ret
        data type string 'float32'

    """
    return current_backend(None).as_ivy_dtype(dtype_in)


def as_native_dtype(dtype_in: Union[ivy.Dtype, ivy.NativeDtype]) -> ivy.NativeDtype:
    """Convert data type string representation to native data type.

    Parameters
    ----------
    dtype_in
        The data type string to convert to native data type.

    Returns
    -------
    ret
        data type e.g. ivy.float32.

    """
    return current_backend(None).as_native_dtype(dtype_in)


# len(get_binary_from_float(x)) >24 and int(get_binary_from_float(x)[24:])>0)
# noinspection PyShadowingBuiltins
def _check_float64(input) -> bool:
    if math.isfinite(input):
        tmp = str(input).replace("-", "").split(".")
        exponent = int(math.floor(math.log10(abs(input)))) if input != 0 else 0
        mant = bin(int(tmp[0])).replace("0b", "")
        return (
            (input > 3.4028235 * 10**38)
            or (len(mant) > 24 and int(mant[24:]) > 0)
            or (exponent < -126)
            or (exponent > 127)
        )
    return False


# noinspection PyShadowingBuiltins
def closest_valid_dtype(type: Union[ivy.Dtype, str, None]) -> Union[ivy.Dtype, str]:
    """Determines the closest valid datatype to the datatype passed as input.

    Parameters
    ----------
    type
        The data type for which to check the closest valid type for.

    Returns
    -------
    ret
        The closest valid data type as a native ivy.Dtype

    """
    return current_backend(type).closest_valid_dtype(type)


# noinspection PyShadowingNames,PyShadowingBuiltins
@handle_nestable
def default_float_dtype(
    input=None,
    float_dtype: Optional[Union[ivy.FloatDtype, ivy.NativeDtype]] = None,
    as_native: Optional[bool] = None,
) -> Union[ivy.Dtype, str]:
    """Summary.

    Parameters
    ----------
    input
         (Default value = None)
    float_dtype

    as_native
         (Default value = None)

    Returns
    -------
        Return the input float dtype if provided, otherwise return the global default
        float dtype.

    """
    if ivy.exists(float_dtype):
        if as_native is True:
            return ivy.as_native_dtype(float_dtype)
        elif as_native is False:
            return ivy.FloatDtype(ivy.as_ivy_dtype(float_dtype))
        return float_dtype
    as_native = ivy.default(as_native, False)
    if ivy.exists(input):
        if ivy.is_native_array(input):
            ret = ivy.dtype(input)
        elif isinstance(input, np.ndarray):
            ret = input.dtype
        elif isinstance(input, (list, tuple, dict)):
            if ivy.nested_indices_where(input, lambda x: _check_float64(x)):
                ret = ivy.float64
            else:
                def_dtype = default_dtype()
                if ivy.is_float_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = ivy.float32
        elif isinstance(input, Number):
            if _check_float64(input):
                ret = ivy.float64
            else:
                def_dtype = default_dtype()
                if ivy.is_float_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = ivy.float32
    else:
        global default_float_dtype_stack
        if not default_float_dtype_stack:
            def_dtype = default_dtype()
            if ivy.is_float_dtype(def_dtype):
                ret = def_dtype
            else:
                ret = "float32"
        else:
            ret = default_float_dtype_stack[-1]
    if as_native:
        return ivy.as_native_dtype(ret)
    return ivy.FloatDtype(ivy.as_ivy_dtype(ret))


# noinspection PyShadowingNames
def default_dtype(
    dtype: Union[ivy.Dtype, str] = None, item=None, as_native: Optional[bool] = None
) -> Union[ivy.Dtype, str]:
    """Summary.

    Parameters
    ----------
    dtype

    item
         (Default value = None)
    as_native
         (Default value = None)

    Returns
    -------
        Return the input dtype if provided, otherwise return the global default dtype.

    """
    if ivy.exists(dtype):
        if as_native is True:
            return ivy.as_native_dtype(dtype)
        elif as_native is False:
            return ivy.as_ivy_dtype(dtype)
        return dtype
    as_native = ivy.default(as_native, False)
    if ivy.exists(item):
        if isinstance(item, (list, tuple, dict)) and len(item) == 0:
            pass
        elif ivy.is_float_dtype(item):
            return default_float_dtype(item, as_native=as_native)
        elif ivy.is_int_dtype(item):
            return default_int_dtype(item, as_native=as_native)
        elif as_native:
            return as_native_dtype("bool")
        else:
            return "bool"
    global default_dtype_stack
    if not default_dtype_stack:
        global default_float_dtype_stack
        if default_float_dtype_stack:
            ret = default_float_dtype_stack[-1]
        else:
            ret = "float32"
    else:
        ret = default_dtype_stack[-1]
    if as_native:
        return ivy.as_native_dtype(ret)
    return ivy.as_ivy_dtype(ret)


# noinspection PyShadowingNames,PyShadowingBuiltins
def default_int_dtype(
    input=None,
    int_dtype: Optional[Union[ivy.IntDtype, ivy.NativeDtype]] = None,
    as_native: Optional[bool] = None,
) -> Union[ivy.IntDtype, ivy.NativeDtype]:
    """Summary.

    Parameters
    ----------
    input
         (Default value = None)
    int_dtype

    as_native
         (Default value = None)

    Returns
    -------
        Return the input int dtype if provided, otherwise return the global default int
        dtype.

    """
    if ivy.exists(int_dtype):
        if as_native is True:
            return ivy.as_native_dtype(int_dtype)
        elif as_native is False:
            return ivy.IntDtype(ivy.as_ivy_dtype(int_dtype))
        return int_dtype
    as_native = ivy.default(as_native, False)
    if ivy.exists(input):
        if ivy.is_native_array(input):
            ret = ivy.dtype(input)
        elif isinstance(input, np.ndarray):
            ret = input.dtype
        elif isinstance(input, (list, tuple, dict)):
            if ivy.nested_indices_where(
                input, lambda x: x > 9223372036854775807 and x != ivy.inf
            ):
                ret = ivy.uint64
            elif ivy.nested_indices_where(
                input, lambda x: x > 2147483647 and x != ivy.inf
            ):
                ret = ivy.int64
            else:
                def_dtype = default_dtype()
                if ivy.is_int_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = ivy.int32
        elif isinstance(input, Number):
            if (
                input > 9223372036854775807
                and input != ivy.inf
                and ivy.backend != "torch"
            ):
                ret = ivy.uint64
            elif input > 2147483647 and input != ivy.inf:
                ret = ivy.int64
            else:
                def_dtype = default_dtype()
                if ivy.is_int_dtype(def_dtype):
                    ret = def_dtype
                else:
                    ret = ivy.int32
    else:
        global default_int_dtype_stack
        if not default_int_dtype_stack:
            def_dtype = default_dtype()
            if ivy.is_int_dtype(def_dtype):
                ret = def_dtype
            else:
                ret = "int32"
        else:
            ret = default_int_dtype_stack[-1]
    if as_native:
        return ivy.as_native_dtype(ret)
    return ivy.IntDtype(ivy.as_ivy_dtype(ret))


def dtype(
    x: Union[ivy.Array, ivy.NativeArray], as_native: bool = False
) -> Union[ivy.Dtype, ivy.NativeDtype]:
    """Get the data type for input array x.

    Parameters
    ----------
    x
        Tensor for which to get the data type.
    as_native
        Whether or not to return the dtype in string format. Default is False.

    Returns
    -------
    ret
        Data type of the array

    """
    return current_backend(x).dtype(x, as_native)


@handle_nestable
def function_supported_dtypes(fn: Callable) -> ivy.Dtype:
    """Returns the supported data types of the current backend's function.

    Parameters
    ----------
    fn
        The function to check for the unsupported dtype attribute

    Returns
    -------
    ret
        The unsupported data types of the function

    Examples
    --------
    >>> ivy.set_backend('torch')
    >>> print(ivy.function_supported_dtypes(ivy.acosh))
    ['int8', 'int16', 'int32', 'int64', 'uint8', \
     'bfloat16', 'float32', 'float64', 'bool']
    """
    valid = list(ivy.valid_dtypes)
    for d in list(function_unsupported_dtypes(fn)):
        if d in valid:
            valid.remove(d)
    return ivy.as_native_dtype(valid)


@handle_nestable
def function_unsupported_dtypes(fn: Callable) -> Tuple:
    """Returns the unsupported data types of the current backend's function.

    Parameters
    ----------
    fn
        The function to check for the unsupported dtype attribute

    Returns
    -------
    ret
        The unsupported data types of the function

    Examples
    --------
    >>> ivy.set_backend('torch')
    >>> print(ivy.function_unsupported_dtypes(ivy.acosh))
    ('float16','uint16','uint32','uint64')

    """
    unsupported_dtypes = ivy.invalid_dtypes
    if hasattr(fn, "unsupported_dtypes"):
        fn_unsupported_dtypes = fn.unsupported_dtypes
        if isinstance(fn_unsupported_dtypes, dict):
            backend_str = ivy.current_backend_str()
            if backend_str in fn_unsupported_dtypes:
                unsupported_dtypes += fn_unsupported_dtypes[backend_str]
            if "all" in fn_unsupported_dtypes:
                unsupported_dtypes += fn_unsupported_dtypes["all"]
        else:
            unsupported_dtypes += fn_unsupported_dtypes
    return tuple(set(unsupported_dtypes))


def invalid_dtype(dtype_in: Union[ivy.Dtype, str, None]) -> bool:
    """Determines whether the provided data type is not support by the current
    framework.

    Parameters
    ----------
    dtype_in
        The data type for which to check for backend non-support

    Returns
    -------
    ret
        Boolean, whether the data-type string is un-supported.

    """
    if dtype_in is None:
        return False
    return ivy.as_ivy_dtype(dtype_in) in ivy.invalid_dtypes


@handle_nestable
@inputs_to_native_arrays
def is_int_dtype(
    dtype_in: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number]
) -> bool:
    """Determine whether the input data type is an int dtype.

    Parameters
    ----------
    dtype_in
        The array or data type to check

    Returns
    -------
    ret
        Whether or not the array or data type is of an integer dtype

    """
    if ivy.is_native_array(dtype_in):
        dtype_in = ivy.dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return "int" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return (
            True
            if isinstance(dtype_in, (int, np.integer))
            and not isinstance(dtype_in, bool)
            else False
        )
    elif isinstance(dtype_in, (list, tuple, dict)):
        return (
            True
            if ivy.nested_indices_where(
                dtype_in,
                lambda x: isinstance(x, (int, np.integer)) and not type(x) == bool,
            )
            else False
        )
    return "int" in as_ivy_dtype(dtype_in)


@inputs_to_native_arrays
@handle_nestable
def is_float_dtype(
    dtype_in: Union[ivy.Dtype, str, ivy.Array, ivy.NativeArray, Number]
) -> bool:
    """Determine whether the input data type is a float dtype.

    Parameters
    ----------
    dtype_in
        The array or data type to check

    Returns
    -------
    ret
        Whether or not the array or data type is of a floating point dtype

    """
    if ivy.is_native_array(dtype_in):
        dtype_in = ivy.dtype(dtype_in)
    elif isinstance(dtype_in, np.ndarray):
        return "float" in dtype_in.dtype.name
    elif isinstance(dtype_in, Number):
        return True if isinstance(dtype_in, (float, np.floating)) else False
    elif isinstance(dtype_in, (list, tuple, dict)):
        return (
            True
            if ivy.nested_indices_where(
                dtype_in, lambda x: isinstance(x, (float, np.floating))
            )
            else False
        )
    return "float" in as_ivy_dtype(dtype_in)


def promote_types(
    type1: Union[ivy.Dtype, ivy.NativeDtype],
    type2: Union[ivy.Dtype, ivy.NativeDtype],
) -> ivy.Dtype:
    """
    Promotes the datatypes type1 and type2, returning the data type they promote to

    Parameters
    ----------
    type1
        the first of the two types to promote
    type2
        the second of the two types to promote

    Returns
    -------
    ret
        The type that both input types promote to
    """
    return ivy.promotion_table[(ivy.as_ivy_dtype(type1), ivy.as_ivy_dtype(type2))]


def set_default_dtype(dtype: Union[ivy.Dtype, str]):
    """Summary.

    Parameters
    ----------
    dtype

    """
    dtype = ivy.as_ivy_dtype(dtype)
    global default_dtype_stack
    default_dtype_stack.append(dtype)


def set_default_float_dtype(float_dtype: Union[ivy.Dtype, str]):
    """Summary.

    Parameters
    ----------
    float_dtype

    """
    float_dtype = ivy.FloatDtype(ivy.as_ivy_dtype(float_dtype))
    global default_float_dtype_stack
    default_float_dtype_stack.append(float_dtype)


def set_default_int_dtype(int_dtype: Union[ivy.Dtype, str]):
    """Summary.

    Parameters
    ----------
    int_dtype

    """
    int_dtype = ivy.IntDtype(ivy.as_ivy_dtype(int_dtype))
    global default_int_dtype_stack
    default_int_dtype_stack.append(int_dtype)


def type_promote_arrays(
    x1: Union[ivy.Array, ivy.NativeArray],
    x2: Union[ivy.Array, ivy.NativeArray],
) -> Tuple:
    """
    Type promote the input arrays, returning new arrays with the shared correct
    data type

    Parameters
    ----------
    x1
        the first of the two arrays to type promote
    x2
        the second of the two arrays to type promote

    Returns
    -------
    ret1, ret2
        The input arrays after type promotion
    """
    new_type = ivy.promote_types(ivy.dtype(x1), ivy.dtype(x2))
    return ivy.astype(x1, new_type), ivy.astype(x2, new_type)


def unset_default_dtype():
    """"""
    global default_dtype_stack
    if default_dtype_stack:
        default_dtype_stack.pop(-1)


# noinspection PyShadowingNames
def unset_default_float_dtype():
    """"""
    global default_float_dtype_stack
    if default_float_dtype_stack:
        default_float_dtype_stack.pop(-1)


# noinspection PyShadowingNames
def unset_default_int_dtype():
    """"""
    global default_int_dtype_stack
    if default_int_dtype_stack:
        default_int_dtype_stack.pop(-1)


def valid_dtype(dtype_in: Union[ivy.Dtype, ivy.NativeDtype, str, None]) -> bool:
    """Determines whether the provided data type is support by the current framework.

    Parameters
    ----------
    dtype_in
        The data type for which to check for backend support

    Returns
    -------
    ret
        Boolean, whether or not the data-type string is supported.

    Examples
    --------
    with :code:`ivy.Dtype` inputs:

    >>> print(ivy.valid_dtype(None))
        True

    >>> print(ivy.valid_dtype(dtype_in = 'float16'))
        True

    >>> print(ivy.valid_dtype('float32'))
        True

    >>> print(ivy.valid_dtype(ivy.float64))
        True

    >>> print(ivy.valid_dtype('bool'))
        True

    >>> print(ivy.valid_dtype(ivy.int8))
        True

    >>> print(ivy.valid_dtype(ivy.int64))
        True

    >>> print(ivy.valid_dtype(ivy.uint8))
        True

    with :code:`ivy.NativeDtype` inputs:

    >>> print(ivy.valid_dtype('native_bool'))
        False

    >>> print(ivy.valid_dtype(ivy.native_float16))
        True

    >>> print(ivy.valid_dtype(ivy.native_float32))
        True

    >>> print(ivy.valid_dtype('native_float64'))
        False

    >>> print(ivy.valid_dtype(ivy.native_int8))
        True

    >>> print(ivy.valid_dtype(ivy.native_int16))
        True

    >>> print(ivy.valid_dtype('native_int32'))
        False

    >>> print(ivy.valid_dtype(ivy.native_int64))
        True

    >>> print(ivy.valid_dtype(ivy.native_uint8))
        True

    >>> print(ivy.valid_dtype('native_uint64'))
        False
    """
    if dtype_in is None:
        return True
    return ivy.as_ivy_dtype(dtype_in) in ivy.valid_dtypes

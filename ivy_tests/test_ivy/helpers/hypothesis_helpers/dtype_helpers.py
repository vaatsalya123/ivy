# global
import numpy as np
from hypothesis import strategies as st
from typing import Optional

# local
import ivy
import ivy.functional.backends.numpy as ivy_np  # ToDo should be removed.
from . import number_helpers as nh
from . import array_helpers as ah
from .. import testing_helpers as th


@st.composite
def array_dtypes(
    draw,
    *,
    num_arrays=st.shared(nh.ints(min_value=1, max_value=4), key="num_arrays"),
    available_dtypes=ivy_np.valid_float_dtypes,
    shared_dtype=False,
    array_api_dtypes=False,
):
    """Draws a list of data types.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    num_arrays
        number of data types to be drawn.
    available_dtypes
        universe of available data types.
    shared_dtype
        if True, all data types in the list are same.
    array_api_dtypes
        if True, use data types that can be promoted with the array_api_promotion
        table.

    Returns
    -------
    A strategy that draws a list.
    """
    if not isinstance(num_arrays, int):
        num_arrays = draw(num_arrays)
    if num_arrays == 1:
        dtypes = draw(ah.list_of_length(x=st.sampled_from(available_dtypes), length=1))
    elif shared_dtype:
        dtypes = draw(ah.list_of_length(x=st.sampled_from(available_dtypes), length=1))
        dtypes = [dtypes[0] for _ in range(num_arrays)]
    else:
        unwanted_types = set(ivy.all_dtypes).difference(set(available_dtypes))
        if array_api_dtypes:
            pairs = ivy.array_api_promotion_table.keys()
        else:
            pairs = ivy.promotion_table.keys()
        available_dtypes = [
            pair for pair in pairs if not any([d in pair for d in unwanted_types])
        ]
        dtypes = list(draw(st.sampled_from(available_dtypes)))
        if num_arrays > 2:
            dtypes += [dtypes[i % 2] for i in range(num_arrays - 2)]
    return dtypes


@st.composite
def get_dtypes(draw, kind, index=0, full=True, none=False, key=None):
    """
    Draws a valid dtypes for the test function. For frontend tests,
    it draws the data types from the intersection between backend
    framework data types and frontend framework dtypes, otherwise,
    draws it from backend framework data types.

    Parameters
    ----------
    draw
        special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    kind
        Supported types are integer, float, valid, numeric, and unsigned
    index
        list indexing incase a test needs to be skipped for a particular dtype(s)
    full
        returns the complete list of valid types
    none
        allow none in the list of valid types

    Returns
    -------
    ret
        dtype string
    """

    def _get_type_dict(framework):
        return {
            "valid": framework.valid_dtypes,
            "numeric": framework.valid_numeric_dtypes,
            "float": framework.valid_float_dtypes,
            "integer": framework.valid_int_dtypes,
            "unsigned": framework.valid_uint_dtypes,
            "signed_integer": tuple(
                set(framework.valid_int_dtypes).difference(framework.valid_uint_dtypes)
            ),
            "complex": framework.valid_complex_dtypes,
            "real_and_complex": tuple(
                set(framework.valid_numeric_dtypes).union(
                    framework.valid_complex_dtypes
                )
            "bool": tuple(
                set(framework.valid_dtypes).difference(framework.valid_numeric_dtypes)
            ),
        }

    backend_dtypes = _get_type_dict(ivy)[kind]
    if th.frontend_fw:
        fw_dtypes = _get_type_dict(th.frontend_fw())[kind]
        valid_dtypes = tuple(set(fw_dtypes).intersection(backend_dtypes))
    else:
        valid_dtypes = backend_dtypes

    if none:
        valid_dtypes += (None,)
    if full:
        return list(valid_dtypes[index:])
    if key is None:
        return [draw(st.sampled_from(valid_dtypes[index:]))]
    return [draw(st.shared(st.sampled_from(valid_dtypes[index:]), key=key))]


@st.composite
def get_castable_dtype(draw, available_dtypes, dtype: str, x: Optional[list] = None):
    """
    Draws castable dtypes for the given dtype based on the current backend.

    Parameters
    ----------
    draw
        Special function that draws data randomly (but is reproducible) from a given
        data-set (ex. list).
    available_dtypes
        Castable data types are drawn from this list randomly.
    dtype
        Data type from which to cast.
    x
        Optional list of values to cast.

    Returns
    -------
    ret
        A tuple of inputs and castable dtype.
    """

    def cast_filter(d):
        if ivy.is_int_dtype(d):
            max_val = ivy.iinfo(d).max
        elif ivy.is_float_dtype(d):
            max_val = ivy.finfo(d).max
        else:
            max_val = 1
        if x is None:
            max_x = -1
        else:
            max_x = np.max(np.abs(np.asarray(x)))
        return max_x <= max_val and ivy.dtype_bits(d) >= ivy.dtype_bits(dtype)

    cast_dtype = draw(st.sampled_from(available_dtypes).filter(cast_filter))
    if x is None:
        return dtype, cast_dtype
    if "uint" in cast_dtype:
        x = np.abs(np.asarray(x)).tolist()
    return dtype, x, cast_dtype

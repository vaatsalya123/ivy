"""Collection of tests for sorting functions."""

# global
from hypothesis import given, strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


# argsort
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        ivy_np.valid_numeric_dtypes,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        min_axis=-1,
        max_axis=0,
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="argsort"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    descending=st.booleans(),
    stable=st.booleans(),
)
def test_argsort(
    dtype_x_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    descending,
    stable,
    fw,
):
    dtype, x, axis = dtype_x_axis

    helpers.test_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "argsort",
        x=np.asarray(x, dtype=dtype),
        axis=axis,
        descending=descending,
        stable=stable,
    )


# sort
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        ivy_np.valid_numeric_dtypes,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        min_axis=-1,
        max_axis=0,
    ),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="sort"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
    descending=st.booleans(),
    stable=st.booleans(),
)
def test_sort(
    dtype_x_axis,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    descending,
    stable,
    fw,
):
    dtype, x, axis = dtype_x_axis

    helpers.test_function(
        dtype,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        container,
        instance_method,
        fw,
        "sort",
        x=np.asarray(x, dtype=dtype),
        axis=axis,
        descending=descending,
        stable=stable,
    )

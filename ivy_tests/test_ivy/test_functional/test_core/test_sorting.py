"""Collection of tests for sorting functions."""

# global
from hypothesis import given, strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# argsort
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        min_axis=-1,
        max_axis=0,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="argsort"),
    descending=st.booleans(),
    stable=st.booleans(),
)
def test_argsort(
    *,
    dtype_x_axis,
    descending,
    stable,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x, axis = dtype_x_axis

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="argsort",
        x=np.asarray(x, dtype=dtype),
        axis=axis,
        descending=descending,
        stable=stable,
    )


# sort
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        min_axis=-1,
        max_axis=0,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="sort"),
    descending=st.booleans(),
    stable=st.booleans(),
)
def test_sort(
    *,
    dtype_x_axis,
    num_positional_args,
    descending,
    stable,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x, axis = dtype_x_axis

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="sort",
        x=np.asarray(x, dtype=dtype),
        axis=axis,
        descending=descending,
        stable=stable,
    )

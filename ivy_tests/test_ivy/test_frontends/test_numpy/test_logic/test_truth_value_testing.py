# global
from hypothesis import given, strategies as st
import ivy
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# all
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("integer"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        min_value=0,
        max_value=1,
        allow_inf=False,
        min_axis=0,
        max_axis=0,
    ),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.all"
    ),
)
def test_numpy_all(
    dtype_x_axis,
    keepdims,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_x_axis
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=[where[0][0]] if isinstance(where, list) else where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="all",
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
        where=where,
        test_values=False,
    )


# any
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("integer"),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=2,
        min_value=0,
        max_value=1,
        allow_inf=False,
        min_axis=0,
        max_axis=0,
    ),
    keepdims=st.booleans(),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.any"
    ),
)
def test_numpy_any(
    dtype_x_axis,
    keepdims,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_x_axis
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=[where[0][0]] if isinstance(where, list) else where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="any",
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
        where=where,
        test_values=False,
    )

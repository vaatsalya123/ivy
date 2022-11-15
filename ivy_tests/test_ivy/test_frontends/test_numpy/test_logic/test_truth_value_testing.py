# global
from hypothesis import strategies as st
import ivy

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# all
@handle_frontend_test(
    fn_tree="numpy.all",
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
)
def test_numpy_all(
    *,
    dtype_x_axis,
    keepdims,
    where,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
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
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
        where=where,
        test_values=False,
    )


# any
@handle_frontend_test(
    fn_tree="numpy.any",
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
)
def test_numpy_any(
    *,
    dtype_x_axis,
    keepdims,
    where,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
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
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        axis=axis,
        out=None,
        keepdims=keepdims,
        where=where,
        test_values=False,
    )


@handle_frontend_test(
    fn_tree="numpy.isscalar",
    element=st.booleans() | st.floats() | st.integers(),
)
def test_numpy_isscalar(
    *,
    element,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=ivy.all_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        element=element,
    )

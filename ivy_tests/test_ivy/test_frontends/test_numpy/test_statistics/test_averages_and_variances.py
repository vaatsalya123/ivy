# global
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    statistical_dtype_values,
)
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy


# mean
@handle_frontend_test(
    fn_tree="numpy.mean",
    dtype_and_x=statistical_dtype_values(function="mean"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_mean(
    dtype_and_x,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtype, x, axis = dtype_and_x
    x_array = ivy.array(x[0])

    if len(x_array.shape) == 2:
        where = ivy.ones((x_array.shape[0], 1), dtype=ivy.bool)
    elif len(x_array.shape) == 1:
        where = True

    if isinstance(axis, tuple):
        axis = axis[0]

    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
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
        x=x[0],
        axis=axis,
        dtype=dtype[0],
        out=None,
        keepdims=keep_dims,
        where=where,
        test_values=False,
    )


# nanmean
@handle_frontend_test(
    fn_tree="numpy.nanmean",
    dtype_and_a=statistical_dtype_values(function="mean"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    keep_dims=st.booleans(),
)
def test_numpy_nanmean(
    dtype_and_a,
    dtype,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
    keep_dims,
):
    input_dtype, a, axis = dtype_and_a
    a_array = ivy.array(a[0])

    if len(a_array.shape) == 2:
        where = ivy.ones((a_array.shape[0], 1), dtype=ivy.bool)
    elif len(a_array.shape) == 1:
        where = True

    if isinstance(axis, tuple):
        axis = axis[0]

    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
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
        a=a[0],
        axis=axis,
        dtype=dtype[0],
        out=None,
        keepdims=keep_dims,
        where=where,
        test_values=False,
    )


# std
@handle_frontend_test(
    fn_tree="numpy.std",
    dtype_and_x=statistical_dtype_values(function="std"),
    dtype=helpers.get_dtypes("float", full=False, none=True),
    where=np_frontend_helpers.where(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.std"
    ),
    keep_dims=st.booleans(),
)
def test_numpy_std(
        dtype_and_x,
        dtype,
        where,
        as_variable,
        with_out,
        num_positional_args,
        native_array,
        fw,
        keep_dims,
):
    input_dtype, x, axis, axis_excess = dtype_and_x
    x_array = ivy.array(x[0])

    if len(x_array.shape) == 2:
        where = ivy.ones((x_array.shape[0], 1), dtype=ivy.bool)
    elif len(x_array.shape) == 1:
        where = True

    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
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
        fn_tree="std",
        x=x[0],
        axis=axis,
        dtype=dtype,
        out=None,
        correction=0,
        keepdims=keep_dims,
        where=where,
        test_values=False,
    )

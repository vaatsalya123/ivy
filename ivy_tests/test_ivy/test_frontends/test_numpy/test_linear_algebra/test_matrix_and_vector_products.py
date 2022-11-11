# global

from hypothesis import given

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
from ivy_tests.test_ivy.test_functional.test_core.test_linalg import (
    _get_first_matrix_and_dtype,
    _get_second_matrix_and_dtype,
    _get_dtype_value1_value2_axis_for_tensordot,
)


# outer
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.outer"
    ),
)
def test_numpy_outer(
    dtype_and_x,
    as_variable,
    with_out,
    native_array,
    num_positional_args,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="outer",
        a=xs[0],
        b=xs[1],
    )


# inner
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-10,
        max_value=10,
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.inner"
    ),
)
def test_numpy_inner(
    dtype_and_x,
    as_variable,
    native_array,
    num_positional_args,
):
    input_dtypes, xs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="inner",
        a=xs[0],
        b=xs[1],
    )


# matmul
@handle_cmd_line_args
@given(
    x=_get_first_matrix_and_dtype(),
    y=_get_second_matrix_and_dtype(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.matmul"
    ),
)
def test_numpy_matmul(
    x, y, as_variable, with_out, native_array, num_positional_args, fw
):
    dtype1, x1 = x
    dtype2, x2 = y
    dtype, dtypes, casting = np_frontend_helpers.handle_dtype_and_casting(
        dtypes=dtype1 + dtype2,
        get_dtypes_kind="numeric",
    )
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="matmul",
        x1=x1,
        x2=x2,
        out=None,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# matrix_power
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        shape=helpers.ints(min_value=2, max_value=8).map(lambda x: tuple([x, x])),
    ),
    n=helpers.ints(min_value=1, max_value=8),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.matrix_power"
    ),
)
def test_numpy_matrix_power(
    dtype_and_x, n, as_variable, native_array, num_positional_args, fw
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="linalg.matrix_power",
        a=x[0],
        n=n,
    )


# tensordot
@handle_cmd_line_args
@given(
    dtype_values_and_axes=_get_dtype_value1_value2_axis_for_tensordot(
        helpers.get_dtypes(kind="numeric")
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.linalg.tensordot"
    ),
)
def test_numpy_tensordot(
    dtype_values_and_axes,
    as_variable,
    native_array,
    num_positional_args,
):
    dtype, a, b, axes = dtype_values_and_axes
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="tensordot",
        a=a,
        b=b,
        axes=axes,
    )

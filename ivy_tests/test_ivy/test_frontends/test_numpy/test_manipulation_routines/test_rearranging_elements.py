from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args
import numpy as np


# roll
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        min_dim_size=2,
    ),
    shift=helpers.ints(min_value=1, max_value=10),
    axis=helpers.ints(min_value=-1, max_value=1),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.roll"
    ),
)
def test_numpy_roll(
    dtype_and_x,
    shift,
    axis,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="roll",
        a=x[0],
        shift=shift,
        axis=axis,
    )


@st.composite
def _dtype_x_bounded_axis(draw, **kwargs):
    dtype, x, shape = draw(helpers.dtype_and_values(**kwargs, ret_shape=True))
    axis = draw(helpers.ints(min_value=0, max_value=len(shape) - 1))
    return dtype, x, axis


@handle_cmd_line_args
@given(
    dtype_x_axis=_dtype_x_bounded_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.flip"
    ),
)
def test_numpy_flip(dtype_x_axis, num_positional_args, native_array):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="flip",
        m=x[0],
        axis=axis,
    )


# flipud
@handle_cmd_line_args
@given(
    dtype_and_m=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-100,
        max_value=100,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.flipud"
    ),
)
def test_numpy_flipud(
    dtype_and_m,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, m = dtype_and_m
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="flipud",
        m=np.asarray(m[0], dtype=input_dtype[0]),
    )

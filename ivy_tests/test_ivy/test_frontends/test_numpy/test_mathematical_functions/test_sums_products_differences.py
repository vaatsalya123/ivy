# global
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _dtype_x_axis(draw, **kwargs):
    dtype, x, shape = draw(helpers.dtype_and_values(**kwargs, ret_shape=True))
    axis = draw(
        st.one_of(
            helpers.ints(min_value=0, max_value=max(len(shape) - 1, 0)), st.none()
        )
    )
    where = draw(
        st.one_of(helpers.array_values(dtype=ivy.bool, shape=shape), st.none())
    )
    return (dtype, x, axis), where


# sum
@handle_cmd_line_args
@given(
    dtype_x_axis=_dtype_x_axis(available_dtypes=ivy_np.valid_float_dtypes),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes + (None,)),
    keep_dims=st.booleans(),
    initial=st.one_of(st.floats(), st.none()),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.sum"
    ),
)
def test_numpy_sum(
    dtype_x_axis,
    dtype,
    keep_dims,
    initial,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    fw,
):
    (input_dtype, x, axis), where = dtype_x_axis
    if initial is None:
        where = True
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="sum",
        x=np.asarray(x, dtype=input_dtype[0]),
        axis=axis,
        dtype=dtype,
        keepdims=keep_dims,
        initial=initial,
        where=where,
    )


# prod
@handle_cmd_line_args
@given(
    dtype_x_axis=_dtype_x_axis(available_dtypes=ivy_np.valid_float_dtypes),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes + (None,)),
    keep_dims=st.booleans(),
    initial=st.one_of(st.floats(), st.none()),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.prod"
    ),
)
def test_numpy_prod(
    dtype_x_axis,
    dtype,
    keep_dims,
    initial,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    fw,
):
    (input_dtype, x, axis), where = dtype_x_axis
    if initial is None:
        where = True
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="prod",
        x=np.asarray(x, dtype=input_dtype[0]),
        axis=axis,
        dtype=dtype,
        keepdims=keep_dims,
        initial=initial,
        where=where,
    )


# cumsum
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("numeric"),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.cumsum"
    ),
)
def test_numpy_cumsum(
    dtype_and_x,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="cumsum",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        dtype=dtype,
    )


# cumprod
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid", full=True),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    dtype=helpers.get_dtypes("numeric"),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.cumprod"
    ),
)
def test_numpy_cumprod(
    dtype_and_x,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x, axis = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="numpy",
        fn_tree="cumprod",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
        dtype=dtype,
    )

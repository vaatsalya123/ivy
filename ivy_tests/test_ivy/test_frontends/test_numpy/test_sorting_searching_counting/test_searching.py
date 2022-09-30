import hypothesis.extra.numpy as hnp
from hypothesis import given, strategies as st
import numpy as np

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def _broadcastable_trio(draw):
    dtype = draw(helpers.get_dtypes("valid", full=False))
    shapes_st = draw(
        hnp.mutually_broadcastable_shapes(num_shapes=3, min_dims=1, min_side=1)
    )
    cond_shape, x1_shape, x2_shape = draw(shapes_st).input_shapes
    cond = draw(hnp.arrays(hnp.boolean_dtypes(), cond_shape))
    x1 = draw(helpers.array_values(dtype=dtype[0], shape=shapes_st))
    x2 = draw(helpers.array_values(dtype=dtype[0], shape=shapes_st))
    return cond, x1, x2, (dtype * 2)


@handle_cmd_line_args
@given(
    broadcastables=_broadcastable_trio(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.where"
    ),
)
def test_numpy_where(
    broadcastables,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    cond, x1, x2, dtype = broadcastables
    helpers.test_frontend_function(
        input_dtypes=["bool"] + dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="where",
        cond=cond,
        x1=x1,
        x2=x2,
    )


@handle_cmd_line_args
@given(
    dtype_and_a=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.nonzero"
    ),
)
def test_numpy_nonzero(
    dtype_and_a,
    native_array,
    num_positional_args,
    fw,
):
    dtype, a = dtype_and_a
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=[False],
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="nonzero",
        a=a[0],
    )


@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.argmin"
    ),
    keep_dims=st.booleans(),
)
def test_numpy_argmin(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="argmin",
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
        out=None,
    )


# argmax
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_axis=-1,
        max_axis=0,
        min_num_dims=1,
        force_int_axis=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.argmax"
    ),
    keep_dims=st.booleans(),
)
def test_numpy_argmax(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    keep_dims,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="argmax",
        a=x[0],
        axis=axis,
        keepdims=keep_dims,
        out=None,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.flatnonzero"
    ),
)
def test_numpy_flatnonzero(
    dtype_and_x, as_variable, native_array, num_positional_args, fw
):
    dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="flatnonzero",
        a=x[0],
    )


@handle_cmd_line_args
@given(
    dtype_x_v=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=1,
        num_arrays=2,
    ),
    side=st.sampled_from(["left", "right"]),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.searchsorted"
    ),
)
def test_numpy_searchsorted(
    dtype_x_v, side, as_variable, native_array, num_positional_args, fw
):
    input_dtypes, xs = dtype_x_v
    helpers.test_frontend_function(
        input_dtypes=input_dtypes + [np.int64],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="searchsorted",
        a=xs[0],
        v=xs[1],
        side=side,
        sorter=np.argsort(xs[0]),
    )

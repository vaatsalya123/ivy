import numpy as np
from hypothesis import given, strategies as st

# local
import ivy.functional.backends.jax as ivy_jax
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_jax.valid_numeric_dtypes,
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.relu"
    ),
)
def test_jax_nn_relu(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.relu",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_jax.valid_numeric_dtypes,
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.relu6"
    ),
)
def test_jax_nn_relu6(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.relu6",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_jax.valid_numeric_dtypes,
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.soft_sign"
    ),
)
def test_jax_nn_soft_sign(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.soft_sign",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_jax.valid_numeric_dtypes,
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.silu"
    ),
)
def test_jax_nn_silu(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.silu",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_jax.valid_numeric_dtypes,
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.leaky_relu"
    ),
    negative_slope=helpers.floats(min_value=0.0, max_value=1.0),
)
def test_jax_nn_leaky_relu(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    negative_slope,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.leaky_relu",
        x=np.asarray(x, dtype=input_dtype),
        negative_slope=negative_slope,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_jax.valid_float_dtypes,  # TODO: use all nums dtypes
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    approximate=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.gelu"
    ),
)
def test_jax_nn_gelu(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
    approximate,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.gelu",
        x=np.asarray(x, dtype=input_dtype),
        approximate=approximate,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_jax.valid_numeric_dtypes,
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.sigmoid"
    ),
)
def test_jax_nn_sigmoid(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.sigmoid",
        x=np.asarray(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_jax.valid_int_dtypes,
        large_value_safety_factor=1,
        small_value_safety_factor=1,
        min_value=1,
        max_value=3,
    ),
    num_classes=st.integers(min_value=4, max_value=6),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.one_hot"
    ),
)
def test_jax_nn_one_hot(
    dtype_and_x,
    num_classes,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.one_hot",
        x=np.asarray(x, dtype=input_dtype),
        num_classes=num_classes,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_jax.valid_float_dtypes,  # TODO: use all float dtypes
        large_value_safety_factor=1,
        small_value_safety_factor=1,
        min_value=-2,
        min_num_dims=1,
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.softmax"
    ),
)
def test_jax_nn_softmax(
    dtype_and_x,
    axis,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.softmax",
        x=np.asarray(x, dtype=input_dtype),
        axis=axis,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_jax.valid_numeric_dtypes,
        large_value_safety_factor=1,
        small_value_safety_factor=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.nn.softplus"
    ),
)
def test_jax_nn_softplus(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="nn.softplus",
        x=np.asarray(x, dtype=input_dtype),
    )

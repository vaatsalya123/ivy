# global
import ivy
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.jax as ivy_jax
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy.valid_numeric_dtypes).intersection(
                set(ivy_jax.valid_numeric_dtypes)
            )
        ),
        min_num_dims=1,
        min_dim_size=1,
    ),
    pred_cond=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.cond"
    ),
)
def test_jax_cond(
    dtype_and_x,
    pred_cond,
    num_positional_args,
    as_variable,
    native_array,
    fw,
):
    def _test_true_fn(x):
        return x + x

    def _test_false_fn(x):
        return x * x

    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=[input_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.cond",
        pred=pred_cond,
        true_fun=_test_true_fn,
        false_fun=_test_false_fn,
        operand=np.array(x, dtype=input_dtype),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy.valid_numeric_dtypes).intersection(
                set(ivy_jax.valid_numeric_dtypes)
            )
        ),
        min_num_dims=1,
        min_dim_size=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.jax.lax.map"
    ),
)
def test_jax_map(
    dtype_and_x,
    num_positional_args,
    as_variable,
    native_array,
    fw,
):
    def _test_map_fn(x):
        return x + x

    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=[input_dtype],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="jax",
        fn_tree="lax.map",
        f=_test_map_fn,
        xs=np.array(x, dtype=input_dtype),
    )

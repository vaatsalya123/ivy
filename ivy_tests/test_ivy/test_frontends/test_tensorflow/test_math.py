# global
import ivy
import numpy as np
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.test_functional.test_core.test_statistical import (
    statistical_dtype_values,
)
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# add
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.add"
    ),
)
def test_tensorflow_add(dtype_and_x, as_variable, num_positional_args, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.add",
        x=x[0],
        y=x[1],
    )


# tan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.tan"
    ),
)
def test_tensorflow_tan(dtype_and_x, as_variable, num_positional_args, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.tan",
        x=x[0],
    )


# multiply
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.multiply"
    ),
)
def test_tensorflow_multiply(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.multiply",
        x=x[0],
        y=x[1],
    )


# maximum
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_maximum(dtype_and_x, as_variable, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=2,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.maximum",
        a=x[0],
        b=x[1],
    )


# subtract
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.subtract"
    ),
)
def test_tensorflow_subtract(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.subtract",
        x=x[0],
        y=x[1],
    )


# logical_xor
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.bool]),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.logical_xor"
    ),
)
def test_tensorflow_logical_xor(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.logical_xor",
        x=x[0],
        y=x[1],
    )


# divide
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.divide"
    ),
)
def test_tensorflow_divide(dtype_and_x, as_variable, num_positional_args, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.divide",
        x=x[0],
        y=x[1],
    )


# negative
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.negative"
    ),
)
def test_tensorflow_negative(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.negative",
        x=x[0],
    )


# logical_and
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.bool]),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.logical_and"
    ),
)
def test_tensorflow_logical_and(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.logical_and",
        x=x[0],
        y=x[1],
    )


# log_sigmoid
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=3,
        small_abs_safety_factor=3,
        safety_factor_scale="linear",
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.log_sigmoid"
    ),
)
def test_tensorflow_log_sigmoid(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.log_sigmoid",
        x=x[0],
    )


# reciprocal_no_nan()
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.reciprocal_no_nan"
    ),
)
def test_tensorflow_reciprocal_no_nan(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.reciprocal_no_nan",
        x=x[0],
    )


# reduce_all()
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.bool]),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.reduce_all"
    ),
)
def test_tensorflow_reduce_all(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.reduce_all",
        input_tensor=x[0],
    )


# reduce_any()
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple([ivy.bool]),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.reduce_any"
    ),
)
def test_tensorflow_reduce_any(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.reduce_any",
        input_tensor=x[0],
    )


# reduce_euclidean_norm()
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.reduce_euclidean_norm"
    ),
)
def test_tensorflow_reduce_euclidean_norm(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    (
        input_dtype,
        x,
    ) = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.reduce_euclidean_norm",
        input_tensor=x[0],
    )


# reduce_logsumexp
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.reduce_logsumexp"
    ),
)
def test_tensorflow_reduce_logsumexp(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.reduce_logsumexp",
        input_tensor=x[0],
    )


# argmax
@handle_cmd_line_args
@given(
    dtype_and_x=statistical_dtype_values(function="argmax"),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.argmax"
    ),
)
def test_tensorflow_argmax(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x, axis = dtype_and_x
    if isinstance(axis, tuple):
        axis = axis[0]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.argmax",
        input=x[0],
        axis=axis,
        output_type="int64",
    )


# reduce_max
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.reduce_max"
    ),
)
def test_tensorflow_reduce_max(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.reduce_max",
        input_tensor=x[0],
    )


# reduce_min
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.reduce_min"
    ),
)
def test_tensorflow_reduce_min(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.reduce_min",
        input_tensor=x[0],
    )


# reduce_prod
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.reduce_prod"
    ),
)
def test_tensorflow_reduce_prod(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.reduce_prod",
        input_tensor=x[0],
    )


# reduce_std
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.reduce_std"
    ),
)
def test_tensorflow_reduce_std(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.reduce_std",
        input_tensor=x[0],
    )


# asinh
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.asinh"
    ),
)
def test_tensorflow_asinh(dtype_and_x, as_variable, num_positional_args, native_array):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.asinh",
        x=x[0],
    )


# reduce_sum
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.reduce_sum"
    ),
)
def test_tensorflow_reduce_sum(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.reduce_sum",
        input_tensor=x[0],
    )


# reduce_mean
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.reduce_mean"
    ),
)
def test_tensorflow_reduce_mean(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.reduce_mean",
        input_tensor=x[0],
    )


# reduce_variance
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.reduce_variance"
    ),
)
def test_tensorflow_reduce_variance(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.reduce_variance",
        input_tensor=x[0],
    )


# scalar_mul
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        min_dim_size=2,
    ),
    scalar_val=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"), shape=(1,)
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.scalar_mul"
    ),
)
def test_tensorflow_scalar_mul(
    dtype_and_x, scalar_val, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    scalar_dtype, scalar = scalar_val
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.scalar_mul",
        scalar=scalar[0][0],
        x=x[0],
    )


# divide_no_nan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        num_arrays=2,
        available_dtypes=helpers.get_dtypes("float"),
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.divide_no_nan"
    ),
)
def test_tensorflow_divide_no_nan(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtypes, xy = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.divide_no_nan",
        x=xy[0],
        y=xy[1],
    )


# multiply_no_nan
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        num_arrays=2,
        available_dtypes=helpers.get_dtypes("float"),
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.multiply_no_nan"
    ),
)
def test_tensorflow_multiply_no_nan(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtypes, xy = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.multiply_no_nan",
        x=xy[0],
        y=xy[1],
    )


# erfcinv
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.erfcinv"
    ),
)
def test_tensorflow_erfcinv(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.erfcinv",
        x=x[0],
    )


# is_non_decreasing
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.is_non_decreasing"
    ),
)
def test_tensorflow_is_non_decreasing(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.is_non_decreasing",
        x=x[0],
    )


# is_strictly_increasing
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.is_strictly_increasing"
    ),
)
def test_tensorflow_is_strictly_increasing(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.is_strictly_increasing",
        x=x[0],
    )


# count_nonzero
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    keepdims=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.count_nonzero"
    ),
    dtype=helpers.get_dtypes("numeric"),
)
def test_tensorflow_count_nonzero(
    dtype_x_axis, dtype, keepdims, as_variable, num_positional_args, native_array
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.count_nonzero",
        input=x,
        axis=axis,
        keepdims=keepdims,
        dtype=dtype,
    )


# confusion_matrix
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=1,
        min_value=0,
        max_value=4,
        shared_dtype=True,
    ),
    num_classes=st.integers(min_value=5, max_value=10),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.confusion_matrix"
    ),
)
def test_tensorflow_confusion_matrix(
    dtype_and_x, num_classes, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.confusion_matrix",
        labels=x[0],
        predictions=x[1],
        num_classes=num_classes,
    )


# polyval
@handle_cmd_line_args
@given(
    dtype_and_coeffs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        max_num_dims=1,
    ),
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
        min_num_dims=0,
        max_num_dims=0,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.polyval"
    ),
)
def test_tensorflow_polyval(
    dtype_and_coeffs, dtype_and_x, as_variable, num_positional_args, native_array
):
    dtype_x, x = dtype_and_x
    dtype_coeffs, coeffs = dtype_and_coeffs
    helpers.test_frontend_function(
        input_dtypes=dtype_coeffs + dtype_x,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.polyval",
        coeffs=coeffs,
        x=x,
    )


# unsorted_segment_mean
@handle_cmd_line_args
@given(
    data=helpers.array_values(dtype=ivy.int32, shape=(5, 6), min_value=1, max_value=9),
    segment_ids=helpers.array_values(
        dtype=ivy.int32, shape=(5,), min_value=0, max_value=4
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.unsorted_segment_mean"
    ),
)
def test_tensorflow_unsorted_segment_mean(
    data, segment_ids, as_variable, num_positional_args, native_array
):
    helpers.test_frontend_function(
        input_dtypes=[ivy.float32, ivy.int32],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.unsorted_segment_mean",
        data=data,
        segment_ids=segment_ids,
        num_segments=np.max(segment_ids) + 1,
    )


# unsorted_segment_sqrt_n
@handle_cmd_line_args
@given(
    data=helpers.array_values(dtype=ivy.int32, shape=(5, 6), min_value=1, max_value=9),
    segment_ids=helpers.array_values(
        dtype=ivy.int32, shape=(5,), min_value=0, max_value=4
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.unsorted_segment_sqrt_n"
    ),
)
def test_tensorflow_unsorted_segment_sqrt_n(
    data, segment_ids, as_variable, num_positional_args, native_array
):
    helpers.test_frontend_function(
        input_dtypes=[ivy.float32, ivy.int32],
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.unsorted_segment_sqrt_n",
        data=data,
        segment_ids=segment_ids,
        num_segments=np.max(segment_ids) + 1,
    )


# zero_fraction
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.zero_fraction"
    ),
)
def test_tensorflow_zero_fraction(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.zero_fraction",
        value=x[0],
    )


# truediv
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.truediv"
    ),
)
def test_tensorflow_truediv(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.truediv",
        x=x[0],
        y=x[1],
    )


# pow
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=[
            "float16",
            "float32",
            "float64",
            "int32",
            "int64",
        ],
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.math.pow"
    ),
)
def test_tensorflow_pow(
    dtype_and_x, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    if x[1].dtype == "int32" or x[1].dtype == "int64":
        if x[1].ndim == 0:
            if x[1] < 0:
                x[1] *= -1
        else:
            x[1][(x[1] < 0).nonzero()] *= -1
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="tensorflow",
        fn_tree="math.pow",
        x=x[0],
        y=x[1],
    )

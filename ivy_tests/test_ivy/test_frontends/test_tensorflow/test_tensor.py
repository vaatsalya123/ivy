# global
import numpy as np
from hypothesis import given, strategies as st

# local
from ivy.functional.frontends.tensorflow import Tensor
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
import ivy.functional.backends.tensorflow as ivy_tf
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# __add__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_add(dtype_and_x, as_variable, native_array, fw):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype[0],
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "y": x[1],
        },
        fw=fw,
        frontend="tensorflow",
        class_name="Tensor",
        method_name="__add__",
    )


# reshape
@st.composite
def dtypes_x_reshape(draw):
    dtypes, x = draw(
        helpers.dtype_and_values(
            shape=helpers.get_shape(
                allow_none=False,
                min_num_dims=1,
                max_num_dims=5,
                min_dim_size=1,
                max_dim_size=10,
            )
        )
    )
    shape = draw(helpers.reshape_shapes(shape=np.array(x).shape))
    return dtypes, x, shape


@handle_cmd_line_args
@given(
    dtypes_x_shape=dtypes_x_reshape(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.tensorflow.Tensor.Reshape",
    ),
)
def test_tensorflow_instance_Reshape(
    dtypes_x_shape,
    as_variable,
    num_positional_args,
    native_array,
    fw,
):
    dtypes, x, shape = dtypes_x_shape
    helpers.test_frontend_array_instance_method(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        fw=fw,
        frontend="tensorflow",
        frontend_class=Tensor,
        fn_tree="Tensor.Reshape",
        self=x[0],
        shape=shape,
    )


# get_shape
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_tensorflow_instance_get_shape(dtype_and_x, as_variable, native_array, fw):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype,
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x,
        },
        input_dtypes_method=[],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={},
        fw=fw,
        frontend="tensorflow",
        class_name="Tensor",
        method_name="get_shape",
    )


# __eq__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
    ),
)
def test_tensorflow_instance_eq(dtype_and_x, as_variable, native_array, fw):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype[0],
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "other": x[1],
        },
        fw=fw,
        frontend="tensorflow",
        class_name="Tensor",
        method_name="__eq__",
    )


# __floordiv__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    )
)
def test_tensorflow_instance_floordiv(dtype_and_x, as_variable, native_array, fw):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype[0],
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "y": x[1],
        },
        fw=fw,
        frontend="tensorflow",
        class_name="Tensor",
        method_name="__floordiv__",
    )


# __ge__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    )
)
def test_tensorflow_instance_ge(dtype_and_x, as_variable, native_array, fw):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype[0],
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "y": x[1],
        },
        fw=fw,
        frontend="tensorflow",
        class_name="Tensor",
        method_name="__ge__",
    )


# __gt__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    )
)
def test_tensorflow_instance_gt(dtype_and_x, as_variable, native_array, fw):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype[0],
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "y": x[1],
        },
        fw=fw,
        frontend="tensorflow",
        class_name="Tensor",
        method_name="__gt__",
    )


# __le__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    )
)
def test_tensorflow_instance_le(dtype_and_x, as_variable, native_array, fw):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype[0],
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "y": x[1],
        },
        fw=fw,
        frontend="tensorflow",
        class_name="Tensor",
        method_name="__le__",
    )


# __lt__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    )
)
def test_tensorflow_instance_lt(dtype_and_x, as_variable, native_array, fw):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype[0],
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "y": x[1],
        },
        fw=fw,
        frontend="tensorflow",
        class_name="Tensor",
        method_name="__lt__",
    )


# __sub__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=tuple(
            set(ivy_np.valid_float_dtypes).intersection(set(ivy_tf.valid_float_dtypes))
        ),
        num_arrays=2,
        shared_dtype=True,
    ),
)
def test_tensorflow_instance_sub(dtype_and_x, as_variable, native_array, fw):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_method(
        input_dtypes_init=input_dtype[0],
        as_variable_flags_init=as_variable,
        num_positional_args_init=0,
        native_array_flags_init=native_array,
        all_as_kwargs_np_init={
            "data": x[0],
        },
        input_dtypes_method=[input_dtype[1]],
        as_variable_flags_method=as_variable,
        num_positional_args_method=0,
        native_array_flags_method=native_array,
        all_as_kwargs_np_method={
            "y": x[1],
        },
        fw=fw,
        frontend="tensorflow",
        class_name="Tensor",
        method_name="__sub__",
    )

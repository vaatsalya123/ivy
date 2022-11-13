from hypothesis import strategies as st
import tensorflow as tf
import sys

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="tensorflow.keras.activations.hard_sigmoid",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_tensorflow_hard_sigmoid(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


@handle_frontend_test(
    fn_tree="tensorflow.keras.activations.linear",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_tensorflow_linear(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# sigmoid
@handle_frontend_test(
    fn_tree="tensorflow.keras.activations.sigmoid",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_tensorflow_sigmoid(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


# tanh
@handle_frontend_test(
    fn_tree="tensorflow.keras.activations.tanh",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_tensorflow_tanh(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


# softmax
@handle_frontend_test(
    fn_tree="tensorflow.keras.activations.softmax",
    dtype_x_and_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_axes_size=1,
        force_int_axis=True,
        valid_axis=True,
    ),
)
def test_tensorflow_softmax(
    *,
    dtype_x_and_axis,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x, axis = dtype_x_and_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        axis=axis,
    )


# gelu test
@handle_frontend_test(
    fn_tree="tensorflow.keras.activations.gelu",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    approximate=st.booleans(),
)
def test_tensorflow_gelu(
    *,
    dtype_and_x,
    approximate,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        approximate=approximate,
    )


@handle_frontend_test(
    fn_tree="tensorflow.keras.activations.elu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
)
def test_tensorflow_relu(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# softplus
@handle_frontend_test(
    fn_tree="tensorflow.keras.activations.softplus",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_tensorflow_softplus(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


# softsign
@handle_frontend_test(
    fn_tree="tensorflow.keras.activations.softsign",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
    ),
)
def test_tensorflow_softsign(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


# swish
@handle_frontend_test(
    fn_tree="tensorflow.keras.activations.swish",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_tensorflow_swish(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        rtol=1e-2,
        atol=1e-2,
        x=x[0],
    )


# elu
@handle_frontend_test(
    fn_tree="tensorflow.keras.activations.elu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-3,
        max_value=3,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
    alpha=st.one_of(
        helpers.floats(
            min_value=-3,
            max_value=3,
        )
    ),
)
def test_tensorflow_elu(
    *,
    dtype_and_x,
    alpha,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        alpha=alpha,
    )


# selu
@handle_frontend_test(
    fn_tree="tensorflow.keras.activations.selu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_value=-3,
        max_value=3,
        min_num_dims=1,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=3,
    ),
)
def test_tensorflow_selu(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
    )


# Helper function for deserialize.
def simple_test_two_function(
    *,
    fn_name: str,
    x,
    frontend: str,
    fn_str: str,
    dtype_data: str,
    rtol_: float = None,
    atol_: float = 1e-06,
    ivy_submodules: list = [],
    framework_submodules: list = [],
):
    ivy.set_backend(frontend)
    fn_ivy = ivy.functional.frontends.__dict__[frontend]
    for ivy_submodule in ivy_submodules:
        fn_ivy = fn_ivy.__dict__[ivy_submodule]
    fn_ivy = fn_ivy.__dict__[fn_str]

    fn_framework = tf
    for framework_submodule in framework_submodules:
        fn_framework = fn_framework.__dict__[framework_submodule]
    fn_framework = fn_framework.__dict__[fn_str]
    x = ivy.array(x).to_native()

    ret_ivy = fn_ivy(fn_name)(x)
    ret = fn_framework(fn_name)(x)

    ret_ivy = ivy.array(ret_ivy, dtype=dtype_data)
    ret = ivy.array(ret, dtype=dtype_data)

    ret_np_flat = helpers.flatten_and_to_np(ret=ret)
    frontend_ret_np_flat = helpers.flatten_and_to_np(ret=ret_ivy)

    helpers.value_test(
        ret_np_flat=ret_np_flat,
        ret_np_from_gt_flat=frontend_ret_np_flat,
        rtol=rtol_,
        atol=atol_,
        ground_truth_backend=frontend,
    )


# Helper function for deserialize.
def get_callable_functions(
    module_name: str,
):
    module = sys.modules[module_name]
    fn_list = list()
    for fn_name in dir(module):
        obj = getattr(module, fn_name)
        if callable(obj):
            fn_list.append(fn_name)
    return fn_list


# deserialize
@handle_frontend_test(
    fn_tree="tensorflow.keras.activations.deserialize",
    fn_name=st.sampled_from(get_callable_functions("keras.activations")).filter(
        lambda x: not x[0].isupper()
        and x
        not in [
            "deserialize",
            "get",
            "keras_export",
            "serialize",
            "deserialize_keras_object",
            "serialize_keras_object",
            "get_globals",
        ]
    ),
    dtype_and_data=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ),
)
def test_tensorflow_deserialize(
    *,
    dtype_and_data,
    fn_name,
    fn_tree,
    frontend,
):
    dtype_data, data = dtype_and_data
    simple_test_two_function(
        fn_name=fn_name,
        x=data[0],
        frontend=frontend,
        fn_tree=fn_tree,
        dtype_data=dtype_data[0],
        rtol_=1e-01,
        atol_=1e-01,
        ivy_submodules=["keras", "activations"],
        framework_submodules=["keras", "activations"],
    )


@handle_frontend_test(
    fn_name=st.sampled_from(get_callable_functions("keras.activations")).filter(
        lambda x: not x[0].isupper()
        and x
        not in [
            "deserialize",
            "get",
            "keras_export",
            "serialize",
            "deserialize_keras_object",
            "serialize_keras_object",
            "get_globals",
        ]
    ),
    dtype_and_data=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ),
)
def test_tensorflow_get(fn_name, dtype_and_data):
    dtype_data, data = dtype_and_data
    simple_test_two_function(
        fn_name=fn_name,
        x=data[0],
        frontend="tensorflow",
        fn_str="get",
        dtype_data=dtype_data[0],
        rtol_=1e-01,
        atol_=1e-01,
        ivy_submodules=["keras", "activations"],
        framework_submodules=["keras", "activations"],
    )

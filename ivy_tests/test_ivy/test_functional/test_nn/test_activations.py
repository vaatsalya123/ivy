"""Collection of tests for unified neural network activation functions."""

# global
from hypothesis import strategies as st, assume

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# relu
@handle_test(
    fn_tree="functional.ivy.relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_relu(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        fw=backend_fw,
        num_positional_args=num_positional_args,
        container_flags=container_flags,
        instance_method=instance_method,
        fn_name=fn_name,
        on_device=on_device,
        test_gradients=test_gradients,
        x=x[0],
    )


# leaky_relu
@handle_test(
    fn_tree="functional.ivy.leaky_relu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=False, key="leaky_relu"),
        large_abs_safety_factor=16,
        small_abs_safety_factor=16,
        safety_factor_scale="log",
    ),
    alpha=st.floats(min_value=-1e-4, max_value=1e-4),
)
def test_leaky_relu(
    *,
    dtype_and_x,
    alpha,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        fw=backend_fw,
        num_positional_args=num_positional_args,
        container_flags=container_flags,
        instance_method=instance_method,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        x=x[0],
        alpha=alpha,
    )


# gelu
@handle_test(
    fn_tree="functional.ivy.gelu",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    approximate=st.booleans(),
)
def test_gelu(
    *,
    dtype_and_x,
    approximate,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        fw=backend_fw,
        num_positional_args=num_positional_args,
        container_flags=container_flags,
        instance_method=instance_method,
        fn_name=fn_name,
        on_device=on_device,
        atol_=1e-2,
        rtol_=1e-2,
        test_gradients=test_gradients,
        x=x[0],
        approximate=approximate,
    )


# sigmoid
@handle_test(
    fn_tree="functional.ivy.sigmoid",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
)
def test_sigmoid(
    *,
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        fw=backend_fw,
        num_positional_args=num_positional_args,
        container_flags=container_flags,
        instance_method=instance_method,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=test_gradients,
        x=x[0],
    )


# softmax
@handle_test(
    fn_tree="functional.ivy.softmax",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
)
def test_softmax(
    *,
    dtype_and_x,
    axis,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        fw=backend_fw,
        num_positional_args=num_positional_args,
        container_flags=container_flags,
        instance_method=instance_method,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        test_gradients=test_gradients,
        x=x[0],
        axis=axis,
    )


# softplus
@handle_test(
    fn_tree="functional.ivy.softplus",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    beta=st.one_of(helpers.number(min_value=0.1, max_value=10), st.none()),
    threshold=st.one_of(helpers.number(min_value=0.1, max_value=30), st.none()),
)
def test_softplus(
    *,
    dtype_and_x,
    beta,
    threshold,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    assume(beta != 0)
    assume(threshold != 0)
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        fw=backend_fw,
        num_positional_args=num_positional_args,
        container_flags=container_flags,
        instance_method=instance_method,
        fn_name=fn_name,
        on_device=on_device,
        test_gradients=test_gradients,
        rtol_=1e-02,
        atol_=1e-02,
        x=x[0],
        beta=beta,
        threshold=threshold,
    )


# log_softmax
@handle_test(
    fn_tree="functional.ivy.log_softmax",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
    ),
    axis=helpers.ints(min_value=-1, max_value=0),
)
def test_log_softmax(
    *,
    dtype_and_x,
    axis,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    test_gradients,
):
    dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        native_array_flags=native_array,
        fw=backend_fw,
        num_positional_args=num_positional_args,
        container_flags=container_flags,
        instance_method=instance_method,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-02,
        atol_=1e-02,
        test_gradients=test_gradients,
        x=x[0],
        axis=axis,
    )

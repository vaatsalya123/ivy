"""Collection of tests for statistical functions."""
# global
import numpy as np
from hypothesis import given, assume, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np


@st.composite
def statistical_dtype_values(draw, *, function):
    dtype = draw(st.sampled_from(ivy_np.valid_float_dtypes))
    size = draw(st.integers(1, 10))
    if dtype == "float16":
        max_value = 2048
    elif dtype == "float32":
        max_value = 16777216
    elif dtype == "float64":
        max_value = 9.0071993e15

    if function == "prod":
        abs_value_limit = 0.99 * max_value ** (1 / size)
    elif function in ["var", "std"]:
        abs_value_limit = 0.99 * (max_value / size) ** 0.5
    else:
        abs_value_limit = 0.99 * max_value / size

    values = draw(
        helpers.list_of_length(
            x=st.floats(
                -abs_value_limit,
                abs_value_limit,
                allow_subnormal=False,
                allow_infinity=False,
            ),
            length=size,
        )
    )
    return dtype, values


# min
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="min"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_min(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    assume(x)
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="min",
        x=np.asarray(x, dtype=input_dtype),
    )


# max
@given(
    dtype_and_x=helpers.dtype_and_values(available_dtypes=ivy_np.valid_numeric_dtypes),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="max"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_max(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    assume(x)
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="max",
        x=np.asarray(x, dtype=input_dtype),
    )


# mean
@given(
    dtype_and_x=statistical_dtype_values(function="mean"),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="mean"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_mean(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="mean",
        test_rtol=1e-1,
        x=np.asarray(x, dtype=input_dtype),
    )


# var
@given(
    dtype_and_x=statistical_dtype_values(function="var"),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="var"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_var(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="var",
        x=np.asarray(x, dtype=input_dtype),
    )


# prod
@given(
    dtype_and_x=statistical_dtype_values(function="prod"),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="prod"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_prod(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    if fw == "torch" and (input_dtype == "float16" or ivy.is_int_dtype(input_dtype)):
        return  # torch implementation exhibits strange behaviour
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="prod",
        x=np.asarray(x, dtype=input_dtype),
    )


# sum
@given(
    dtype_and_x=statistical_dtype_values(function="sum"),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="sum"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_sum(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="sum",
        test_rtol=1e-2,
        x=np.asarray(x, dtype=input_dtype),
    )


# std
@given(
    dtype_and_x=statistical_dtype_values(function="std"),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="std"),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_std(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="std",
        test_rtol=1e-2,
        test_atol=1e-2,
        x=np.asarray(x, dtype=input_dtype),
    )


# einsum
@given(
    eq_n_op_n_shp=st.sampled_from(
        [
            ("ii", (np.arange(25).reshape(5, 5),), ()),
            ("ii->i", (np.arange(25).reshape(5, 5),), (5,)),
            ("ij,j", (np.arange(25).reshape(5, 5), np.arange(5)), (5,)),
        ]
    ),
    dtype=st.sampled_from(ivy_np.valid_float_dtypes),
    with_out=st.booleans(),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
)
def test_einsum(eq_n_op_n_shp, dtype, with_out, tensor_fn, device, call):
    # smoke test
    eq, operands, true_shape = eq_n_op_n_shp
    operands = [tensor_fn(op, dtype=dtype, device=device) for op in operands]
    if with_out:
        out = ivy.zeros(true_shape, dtype=dtype)
        ret = ivy.einsum(eq, *operands, out=out)
    else:
        ret = ivy.einsum(eq, *operands)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == true_shape
    # value test
    assert np.allclose(
        call(ivy.einsum, eq, *operands),
        ivy.functional.backends.numpy.einsum(
            eq, *[ivy.to_numpy(op) for op in operands]
        ),
    )
    # out test
    if with_out:
        assert ret is out
        if ivy.current_backend_str() in ["tensorflow", "jax"]:
            # these backends do not support native inplace updates
            return
        assert ret.data is out.data

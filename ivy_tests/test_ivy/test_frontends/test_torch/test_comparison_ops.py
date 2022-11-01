# global
import numpy as np
from hypothesis import given, assume, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# allclose
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.allclose"
    ),
    equal_nan=st.booleans(),
)
def test_torch_allclose(
    dtype_and_input,
    equal_nan,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="allclose",
        rtol=1e-05,
        atol=1e-08,
        input=input[0],
        other=input[1],
        equal_nan=equal_nan,
    )


# equal
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.equal"
    ),
)
def test_torch_equal(
    dtype_and_inputs,
    as_variable,
    num_positional_args,
    native_array,
):
    inputs_dtypes, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=inputs_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="equal",
        input=inputs[0],
        other=inputs[1],
    )


# eq
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.eq"
    ),
)
def test_torch_eq(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    inputs_dtypes, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=inputs_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="eq",
        input=inputs[0],
        other=inputs[1],
        out=None,
    )


# argsort
@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        min_axis=-1,
        max_axis=0,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.argsort"
    ),
    descending=st.booleans(),
)
def test_torch_argsort(
    dtype_input_axis,
    descending,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="argsort",
        input=input[0],
        dim=axis,
        descending=descending,
    )


# greater_equal
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.greater_equal"
    ),
)
def test_torch_greater_equal(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        all_aliases=["ge"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="greater_equal",
        input=inputs[0],
        other=inputs[1],
        out=None,
    )


# greater
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        allow_inf=False,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.greater"
    ),
)
def test_torch_greater(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        all_aliases=["gt"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="greater",
        input=inputs[0],
        other=inputs[1],
        out=None,
    )


# isclose
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.isclose"
    ),
    equal_nan=st.booleans(),
)
def test_torch_isclose(
    dtype_and_input,
    equal_nan,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="isclose",
        rtol=1e-05,
        atol=1e-08,
        input=input[0],
        other=input[1],
        equal_nan=equal_nan,
    )


# isfinite
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.isfinite"
    ),
)
def test_torch_isfinite(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="isfinite",
        input=input[0],
    )


# isinf
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.isinf"
    ),
)
def test_torch_isinf(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="isinf",
        input=input[0],
    )


# isposinf
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.isposinf"
    ),
)
def test_torch_isposinf(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="isposinf",
        input=input[0],
    )


# isneginf
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.isneginf"
    ),
)
def test_torch_isneginf(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="isneginf",
        input=input[0],
    )


# sort
@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        min_dim_size=1,
        min_axis=-1,
        max_axis=0,
    ),
    descending=st.booleans(),
    stable=st.booleans(),
)
def test_torch_sort(
    dtype_input_axis,
    descending,
    stable,
    as_variable,
    with_out,
    native_array,
):
    input_dtype, input, axis = dtype_input_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="sort",
        input=input[0],
        dim=axis,
        descending=descending,
        stable=stable,
        out=None,
    )


# isnan
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_value=-np.inf,
        max_value=np.inf,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.isnan"
    ),
)
def test_torch_isnan(
    dtype_and_input,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="isnan",
        input=input[0],
    )


# less_equal
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.less_equal"
    ),
)
def test_torch_less_equal(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        all_aliases=["le"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="less_equal",
        input=inputs[0],
        other=inputs[1],
        out=None,
    )


# less
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.less"
    ),
)
def test_torch_less(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        all_aliases=["lt"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="less",
        input=inputs[0],
        other=inputs[1],
        out=None,
    )


# not_equal
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.not_equal"
    ),
)
def test_torch_not_equal(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        all_aliases=["ne"],
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="not_equal",
        input=inputs[0],
        other=inputs[1],
        out=None,
    )


@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.isin"
    ),
)
def test_torch_isin(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="isin",
        elements=inputs[0],
        test_elements=inputs[1],
    )


@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.minimum"
    ),
)
def test_torch_minimum(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="minimum",
        input=inputs[0],
        other=inputs[1],
        out=None,
    )


# fmax
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-np.inf,
        max_value=np.inf,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.fmax"
    ),
)
def test_torch_fmax(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="fmax",
        input=inputs[0],
        other=inputs[1],
        out=None,
    )


# fmin
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
        min_value=-np.inf,
        max_value=np.inf,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.fmin"
    ),
)
def test_torch_fmin(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="fmin",
        input=inputs[0],
        other=inputs[1],
        out=None,
    )


# msort
@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,
        min_dim_size=2,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.msort"
    ),
)
def test_torch_msort(
    dtype_and_input,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="msort",
        input=input[0],
    )


# maximum
@handle_cmd_line_args
@given(
    dtype_and_inputs=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.maximum"
    ),
)
def test_torch_maximum(
    dtype_and_inputs,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, inputs = dtype_and_inputs
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="maximum",
        input=np.asarray(inputs[0], dtype=input_dtype[0]),
        other=np.asarray(inputs[1], dtype=input_dtype[1]),
        out=None,
    )


# kthvalue
@handle_cmd_line_args
@given(
    dtype_input_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        valid_axis=True,
        force_int_axis=True,
    ),
    k=st.integers(min_value=1),
    keepdim=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.kthvalue"
    ),
)
def test_torch_kthvalue(
    dtype_input_axis,
    k,
    keepdim,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, input, dim = dtype_input_axis
    assume(k <= input[0].shape[dim])
    assume("float16" not in input_dtype)  # unsupported by torch.kthvalue
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="kthvalue",
        input=input[0],
        k=k,
        dim=dim,
        keepdim=keepdim,
        out=None,
    )


# topk
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
        large_abs_safety_factor=8,
        small_abs_safety_factor=8,
        safety_factor_scale="log",
        min_dim_size=4,
        max_dim_size=10,
    ),
    dim=helpers.ints(min_value=-1, max_value=0),
    k=helpers.ints(min_value=1, max_value=4),
    largest=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.topk"
    ),
)
def test_torch_topk(
    dtype_and_x,
    k,
    dim,
    largest,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, input = dtype_and_x
    assume("float16" not in input_dtype)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="topk",
        input=input[0],
        k=k,
        dim=dim,
        largest=largest,
        out=None,
    )

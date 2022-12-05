# global
import ivy
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


# Helper functions


@st.composite
def _fill_value(draw):
    dtype = draw(st.shared(helpers.get_dtypes("numeric", full=False), key="dtype"))[0]
    if ivy.is_uint_dtype(dtype):
        return draw(helpers.ints(min_value=0, max_value=5))
    elif ivy.is_int_dtype(dtype):
        return draw(helpers.ints(min_value=-5, max_value=5))
    return draw(helpers.floats(min_value=-5, max_value=5))


@st.composite
def _start_stop_step(draw):
    start = draw(helpers.ints(min_value=0, max_value=50))
    stop = draw(helpers.ints(min_value=0, max_value=50))
    if start < stop:
        step = draw(helpers.ints(min_value=1, max_value=50))
    else:
        step = draw(helpers.ints(min_value=-50, max_value=-1))
    return start, stop, step


# full
@handle_frontend_test(
    fn_tree="torch.full",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    fill_value=_fill_value(),
    dtype=st.shared(helpers.get_dtypes("numeric", full=False), key="dtype"),
)
def test_torch_full(
    *,
    shape,
    fill_value,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        on_device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        size=shape,
        fill_value=fill_value,
        dtype=dtype[0],
        device=on_device,
    )


# ones_like
@handle_frontend_test(
    fn_tree="torch.ones_like",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    dtype=helpers.get_dtypes("numeric", full=False),
)
def test_torch_ones_like(
    *,
    dtype_and_x,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        dtype=dtype[0],
        device=on_device,
    )


# ones
@handle_frontend_test(
    fn_tree="torch.ones",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
)
def test_torch_ones(
    *,
    shape,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        size=shape,
        dtype=dtype[0],
        device=on_device,
    )


# zeros
@handle_frontend_test(
    fn_tree="torch.zeros",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("numeric", full=False),
)
def test_torch_zeros(
    *,
    shape,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        size=shape,
        dtype=dtype[0],
        device=on_device,
    )


# zeros_like
@handle_frontend_test(
    fn_tree="torch.zeros_like",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    dtype=helpers.get_dtypes("numeric", full=False),
)
def test_torch_zeros_like(
    *,
    dtype_and_x,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        input=input[0],
        dtype=dtype[0],
        device=on_device,
    )


# empty
@handle_frontend_test(
    fn_tree="torch.empty",
    shape=helpers.get_shape(
        allow_none=False,
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=10,
    ),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_torch_empty(
    *,
    shape,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        size=shape,
        dtype=dtype[0],
        test_values=False,
        device=on_device,
    )


# arange
@handle_frontend_test(
    fn_tree="torch.arange",
    start_stop_step=_start_stop_step(),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_torch_arange(
    *,
    start_stop_step,
    dtype,
    as_variable,
    with_out,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    start, stop, step = start_stop_step
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=3,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        end=stop,
        step=step,
        dtype=dtype[0],
        device=on_device,
    )


# range
@handle_frontend_test(
    fn_tree="torch.range",
    start_stop_step=_start_stop_step(),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_torch_range(
    *,
    start_stop_step,
    dtype,
    as_variable,
    with_out,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    start, stop, step = start_stop_step
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=3,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        end=stop,
        step=step,
        dtype=dtype[0],
        device=on_device,
    )


# linspace
@handle_frontend_test(
    fn_tree="torch.linspace",
    start=st.floats(min_value=-10, max_value=10),
    stop=st.floats(min_value=-10, max_value=10),
    num=st.integers(min_value=1, max_value=10),
)
def test_torch_linspace(
    *,
    start,
    stop,
    num,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        end=stop,
        steps=num,
        device=on_device,
    )


# logspace
@handle_frontend_test(
    fn_tree="torch.logspace",
    start=st.floats(min_value=-10, max_value=10),
    stop=st.floats(min_value=-10, max_value=10),
    num=st.integers(min_value=1, max_value=10),
)
def test_torch_logspace(
    *,
    start,
    stop,
    num,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    helpers.test_frontend_function(
        input_dtypes=[],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        start=start,
        end=stop,
        steps=num,
        device=on_device,
    )


# empty_like
@handle_frontend_test(
    fn_tree="torch.empty_like",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_torch_empty_like(
    *,
    dtype_and_x,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, inputs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        input=inputs[0],
        dtype=dtype[0],
        device=on_device,
        test_values=False,
    )


# full_like
@handle_frontend_test(
    fn_tree="torch.full_like",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=st.shared(
            helpers.get_dtypes("numeric", full=False), key="dtype"
        )
    ),
    fill_value=_fill_value(),
    dtype=st.shared(helpers.get_dtypes("numeric", full=False), key="dtype"),
)
def test_torch_full_like(
    *,
    dtype_and_x,
    fill_value,
    dtype,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    input_dtype, inputs = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        input=inputs[0],
        fill_value=fill_value,
        dtype=dtype[0],
        on_device=on_device,
        test_values=False,
    )


# as_tensor
@handle_frontend_test(
    fn_tree="torch.as_tensor",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_torch_as_tensor(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        data=input[0],
        device=on_device,
    )


# from_numpy
@handle_frontend_test(
    fn_tree="torch.from_numpy",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_torch_from_numpy(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        on_device=on_device,
        frontend=frontend,
        fn_tree=fn_tree,
        data=input[0],
    )


# tensor
@handle_frontend_test(
    fn_tree="torch.tensor",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("valid")),
)
def test_torch_tensor(
    *,
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    on_device,
    fn_tree,
    frontend,
):
    dtype, input = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        data=input[0],
        device=on_device,
    )

# global
from hypothesis import given, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# noinspection DuplicatedCode
@st.composite
def _arrays_idx_n_dtypes(draw):
    num_dims = draw(st.shared(helpers.ints(min_value=1, max_value=4), key="num_dims"))
    num_arrays = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="num_arrays")
    )
    common_shape = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_dims - 1,
            max_size=num_dims - 1,
        )
    )
    unique_idx = draw(helpers.ints(min_value=0, max_value=num_dims - 1))
    unique_dims = draw(
        helpers.lists(
            arg=helpers.ints(min_value=2, max_value=3),
            min_size=num_arrays,
            max_size=num_arrays,
        )
    )
    xs = list()
    input_dtypes = draw(
        helpers.array_dtypes(available_dtypes=draw(helpers.get_dtypes("valid")))
    )
    for ud, dt in zip(unique_dims, input_dtypes):
        x = draw(
            helpers.array_values(
                shape=common_shape[:unique_idx] + [ud] + common_shape[unique_idx:],
                dtype=dt,
            )
        )
        xs.append(x)
    return xs, input_dtypes, unique_idx


# concat
@handle_cmd_line_args
@given(
    xs_n_input_dtypes_n_unique_idx=_arrays_idx_n_dtypes(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.concatenate"
    ),
)
def test_numpy_concatenate(
    xs_n_input_dtypes_n_unique_idx,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    xs, input_dtypes, unique_idx = xs_n_input_dtypes_n_unique_idx
    dtype, input_dtypes, casting = np_frontend_helpers.handle_dtype_and_casting(
        dtypes=input_dtypes,
        get_dtypes_kind="valid",
    )
    helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="concatenate",
        arrays=xs,
        axis=unique_idx,
        out=None,
        dtype=dtype,
        casting=casting,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    factor=helpers.ints(min_value=2, max_value=6),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.stack"
    ),
)
def test_numpy_stack(
    dtype_and_x,
    factor,
    as_variable,
    native_array,
    with_out,
    num_positional_args,
):
    dtype, x = dtype_and_x
    xs = [x[0]]
    for i in range(factor):
        xs += [x[0]]
    helpers.test_frontend_function(
        input_dtypes=[dtype[0]] * (factor + 1),
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="stack",
        arrays=xs,
        axis=0,
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    factor=helpers.ints(min_value=2, max_value=6),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.vstack"
    ),
)
def test_numpy_vstack(
    dtype_and_x,
    factor,
    as_variable,
    native_array,
    with_out,
    num_positional_args,
):
    dtype, x = dtype_and_x
    xs = [x[0]]
    for i in range(factor):
        xs += [x[0]]
    helpers.test_frontend_function(
        input_dtypes=[dtype[0]] * (factor + 1),
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="vstack",
        tup=xs
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=1,
    ),
    factor=helpers.ints(min_value=2, max_value=6),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.numpy.hstack"
    ),
)
def test_numpy_hstack(
    dtype_and_x,
    factor,
    as_variable,
    native_array,
    with_out,
    num_positional_args,
):
    dtype, x = dtype_and_x
    xs = [x[0], ]
    for i in range(factor):
        xs += [x[0], ]
    helpers.test_frontend_function(
        input_dtypes=[dtype[0]] * (factor + 1),
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="numpy",
        fn_tree="hstack",
        tup=xs
    )

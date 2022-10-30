# global
import math

import numpy as np
from hypothesis import assume, given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


# helpers
@st.composite
def _get_repeat_interleaves_args(
    draw, *, available_dtypes, valid_axis, max_num_dims, max_dim_size
):
    values_dtype, values, axis, shape = draw(
        helpers.dtype_values_axis(
            available_dtypes=available_dtypes,
            valid_axis=valid_axis,
            force_int_axis=True,
            shape=draw(
                helpers.get_shape(
                    allow_none=False,
                    min_num_dims=0,
                    max_num_dims=max_num_dims,
                    min_dim_size=1,
                    max_dim_size=max_dim_size,
                )
            ),
            ret_shape=True,
        )
    )

    if axis is None:
        generate_repeats_as_integer = draw(st.booleans())
        num_repeats = 1 if generate_repeats_as_integer else math.prod(tuple(shape))
    else:
        num_repeats = shape[axis]

    repeats_dtype, repeats = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("integer"),
            min_value=0,
            max_value=10,
            shape=[num_repeats],
        )
    )

    # Output size is an optional parameter accepted by Torch for optimisation
    use_output_size = draw(st.booleans())
    output_size = np.sum(repeats) if use_output_size else None

    return [values_dtype, repeats_dtype], values, repeats, axis, output_size


# flip
@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        available_dtypes=helpers.get_dtypes("float"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
        force_tuple=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.flip"
    ),
)
def test_torch_flip(
    dtype_and_values,
    axis,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="flip",
        input=value[0],
        dims=axis,
    )


# roll
@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    shift=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    axis=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=1), key="shape"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.roll"
    ),
)
def test_torch_roll(
    dtype_and_values,
    shift,
    axis,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, value = dtype_and_values
    if isinstance(shift, int) and isinstance(axis, tuple):
        axis = axis[0]
    if isinstance(shift, tuple) and isinstance(axis, tuple):
        if len(shift) != len(axis):
            mn = min(len(shift), len(axis))
            shift = shift[:mn]
            axis = axis[:mn]
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="roll",
        input=value[0],
        shifts=shift,
        dims=axis,
    )


# fliplr
@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=helpers.get_shape(min_num_dims=2),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.fliplr"
    ),
)
def test_torch_fliplr(
    dtype_and_values,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="fliplr",
        input=value[0],
    )


# cumsum
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.cumsum"
    ),
    dtype=helpers.get_dtypes("numeric", none=True),
)
def test_torch_cumsum(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    dtype,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="cumsum",
        input=x[0],
        dim=axis,
        dtype=dtype,
        out=None,
    )


@st.composite
def dims_and_offset(draw, shape):
    shape_actual = draw(shape)
    dim1 = draw(helpers.get_axis(shape=shape, force_int=True))
    dim2 = draw(helpers.get_axis(shape=shape, force_int=True))
    offset = draw(
        st.integers(min_value=-shape_actual[dim1], max_value=shape_actual[dim1])
    )
    return dim1, dim2, offset


@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
    ),
    dims_and_offset=dims_and_offset(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape")
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.diagonal"
    ),
)
def test_torch_diagonal(
    dtype_and_values,
    dims_and_offset,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, value = dtype_and_values
    dim1, dim2, offset = dims_and_offset
    input = value[0]
    num_dims = len(np.shape(input))
    assume(dim1 != dim2)
    if dim1 < 0:
        assume(dim1 + num_dims != dim2)
    if dim2 < 0:
        assume(dim1 != dim2 + num_dims)
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="diagonal",
        input=input,
        offset=offset,
        dim1=dim1,
        dim2=dim2,
    )


@handle_cmd_line_args
@given(
    dtype_and_tensors=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=st.integers(min_value=1, max_value=5),
        min_num_dims=1,
        max_num_dims=1,
        max_dim_size=5,
        shared_dtype=True,
    )
)
def test_torch_cartesian_prod(
    dtype_and_tensors,
    as_variable,
    native_array,
    with_out,
):
    dtypes, tensors = dtype_and_tensors
    if isinstance(dtypes, list):  # If more than one value was generated
        args = {
            f"x{i}": np.array(tensor, dtype=dtypes[i])
            for i, tensor in enumerate(tensors)
        }
    else:  # If exactly one value was generated
        args = {"x0": np.array(tensors, dtype=dtypes)}
    num_positional_args = len(tensors)
    helpers.test_frontend_function(
        input_dtypes=dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="cartesian_prod",
        **args,
    )


@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=2,  # Torch requires this.
    ),
    diagonal=st.integers(min_value=-100, max_value=100),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.triu"
    ),
)
def test_torch_triu(
    dtype_and_values,
    diagonal,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
):
    dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="triu",
        input=values[0],
        diagonal=diagonal,
    )


# cumprod
@handle_cmd_line_args
@given(
    dtype_x_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        valid_axis=True,
        allow_neg_axes=False,
        max_axes_size=1,
        force_int_axis=True,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.cumprod"
    ),
    dtype=helpers.get_dtypes("numeric", none=True),
)
def test_torch_cumprod(
    dtype_x_axis,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    dtype,
):
    input_dtype, x, axis = dtype_x_axis
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="cumprod",
        input=x[0],
        dim=axis,
        dtype=dtype,
        out=None,
    )


# trace
@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        shape=st.shared(helpers.get_shape(min_num_dims=2, max_num_dims=2), key="shape"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.trace"
    ),
)
def test_torch_trace(
    dtype_and_values,
    as_variable,
    num_positional_args,
    with_out,
    native_array,
):
    dtype, value = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="trace",
        input=value[0],
    )


@handle_cmd_line_args
@given(
    row=st.integers(min_value=0, max_value=10),
    col=st.integers(min_value=0, max_value=10),
    offset=st.integers(),
    dtype_result=helpers.get_dtypes("valid"),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.tril_indices"
    ),
)
def test_torch_tril_indices(
    row,
    col,
    offset,
    dtype_result,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    helpers.test_frontend_function(
        input_dtypes=[ivy.int32],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="tril_indices",
        row=row,
        col=col,
        offset=offset,
        dtype=dtype_result,
    )


@handle_cmd_line_args
@given(
    row=st.integers(min_value=0, max_value=100),
    col=st.integers(min_value=0, max_value=100),
    offset=st.integers(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.triu_indices"
    ),
)
def test_torch_triu_indices(
    row,
    col,
    offset,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    helpers.test_frontend_function(
        input_dtypes=["int32"],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="triu_indices",
        row=row,
        col=col,
        offset=offset,
    )


@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=2,  # Torch requires this.
    ),
    diagonal=st.integers(min_value=-100, max_value=100),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.tril"
    ),
)
def test_torch_tril(
    dtype_and_values,
    diagonal,
    num_positional_args,
    as_variable,
    with_out,
    native_array,
):
    dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="tril",
        input=values[0],
        diagonal=diagonal,
    )


@st.composite
def _get_dtype_and_arrays_and_start_end_dim(
    draw,
    *,
    available_dtypes,
    min_num_dims=1,
    max_num_dims=5,
    min_dim_size=1,
    max_dim_size=5,
):
    """Samples a dtype, array, and start and end dimension which are within the array,
    with the caveat that the end dimension can be `-1`. This is to match the API
    for PyTorch's flatten.

    Parameters
    ----------
    available_dtypes
        The dtypes that are permitted for the array, expected to be
        `helpers.get_dtypes("valid") or similar.

    min_num_dims
        The minimum number of dimensions the array can have. Defaults to 1

    max_num_dims
        The maximum number of dimensions the array can have. Defaults to 5

    min_dim_size
        The minimum size of any dimension in the array. Defaults to 1

    max_dim_size
        The maximum size of any dimension in the array. Defaults to 5

    Returns
    -------
    ret
        A 4-tuple (dtype, array, start_dim, end_dim) where dtype is
        one of the available dtypes, the array is an array of values
        and start_dim and end_dim are legal dimensions contained
        within the array, with either start_dim <= end_dim or
        end_dim = 1.

    """
    num_dims = draw(st.integers(min_value=min_num_dims, max_value=max_num_dims))
    shape = tuple(
        draw(st.integers(min_value=min_dim_size, max_value=max_dim_size))
        for _ in range(num_dims)
    )

    dtype, array = draw(
        helpers.dtype_and_values(
            available_dtypes=available_dtypes,
            shape=shape,
        )
    )

    start_dim = draw(st.integers(min_value=0, max_value=num_dims - 1))

    # End_dim must be either -1 or in [start_dim, num_dims)
    # If end_dim is -1, then it's going to flatten to a 1-D array.
    is_full_flatten = draw(st.booleans())
    if is_full_flatten:
        end_dim = -1
    else:
        end_dim = draw(st.integers(min_value=start_dim, max_value=num_dims - 1))

    return dtype, array, start_dim, end_dim


@handle_cmd_line_args
@given(
    dtype_and_input_and_start_end_dim=_get_dtype_and_arrays_and_start_end_dim(
        available_dtypes=helpers.get_dtypes("valid"),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.flatten"
    ),
)
def test_torch_flatten(
    dtype_and_input_and_start_end_dim,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    dtype, input, start_dim, end_dim = dtype_and_input_and_start_end_dim
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="flatten",
        input=input[0],
        start_dim=start_dim,
        end_dim=end_dim,
    )


@handle_cmd_line_args
@given(
    dtype_and_values=helpers.dtype_and_values(
        # Min_num_dims is 2 to prevent a Torch crash.
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
        # Setting available types to valid allows Bool and integer types
        # which causes a Torch crash.
        available_dtypes=helpers.get_dtypes("float"),
        max_value=1e4,
        min_value=-1e4,
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"),
        force_int=True,
    ),
    p=st.floats(
        min_value=0.5,
        exclude_min=True,
        max_value=5,
    ),  # Non-positive norms aren't supported in backends.
    # Small positive norms cause issues due to finite-precision.
    maxnorm=st.floats(min_value=0),  # Norms are positive semi-definite
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.renorm"
    ),
)
def test_torch_renorm(
    dtype_and_values,
    p,
    dim,
    maxnorm,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    dtype, values = dtype_and_values
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="renorm",
        atol=1e-02,
        input=values[0],
        p=p,
        dim=dim,
        maxnorm=maxnorm,
    )


@handle_cmd_line_args
@given(
    dtype_and_input=helpers.dtype_and_values(
        # Torch version is not implemented for Integer or Bool types
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.shared(helpers.get_shape(), key="shape"),
        max_value=100,
        min_value=-100,
    ),
    dim=helpers.get_axis(
        shape=st.shared(helpers.get_shape(), key="shape"), force_int=True
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.logcumsumexp"
    ),
)
def test_torch_logcumsumexp(
    dtype_and_input,
    dim,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    dtype, input = dtype_and_input
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="logcumsumexp",
        rtol=1e-2,
        atol=1e-2,
        input=input[0],
        dim=dim,
    )


@handle_cmd_line_args
@given(
    dtype_values_repeats_axis_output_size=_get_repeat_interleaves_args(
        available_dtypes=helpers.get_dtypes("valid"),
        valid_axis=True,
        max_num_dims=4,
        max_dim_size=4,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.repeat_interleave",
    ),
)
def test_torch_repeat_interleave(
    dtype_values_repeats_axis_output_size,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    dtype, values, repeats, axis, output_size = dtype_values_repeats_axis_output_size

    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="repeat_interleave",
        input=values,
        repeats=repeats,
        dim=axis,
        output_size=output_size,
    )


# ravel
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=1,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.ravel"
    ),
)
def test_torch_ravel(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="ravel",
        input=np.asarray(x[0], dtype=input_dtype[0]),
    )


# rot90
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
    dims=helpers.get_axis(
        shape=st.shared(helpers.get_shape(min_num_dims=2), key="shape"),
        min_size=2,
        max_size=2,
        unique=True,
        allow_neg=False,
        force_tuple=True,
    ),
    k=st.integers(min_value=-10, max_value=10),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.rot90"
    ),
)
def test_torch_rot90(
    dtype_and_x, dims, k, as_variable, num_positional_args, native_array
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="rot90",
        input=x[0],
        k=k,
        dims=dims,
    )


# vander
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        shape=st.tuples(
            st.integers(min_value=1, max_value=5),
        ),
    ),
    N=st.integers(min_value=0, max_value=5),
    increasing=st.booleans(),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.vander"
    ),
)
def test_torch_vander(
    dtype_and_x,
    N,
    increasing,
    as_variable,
    num_positional_args,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="vander",
        x=np.asarray(x[0], dtype=input_dtype[0]),
        N=N,
        increasing=increasing,
    )


# lcm
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("integer"),
        num_arrays=2,
        shared_dtype=True,
        min_num_dims=1,
        max_num_dims=3,
        min_value=-100,
        max_value=100,
        allow_nan=False,
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="ivy.functional.frontends.torch.lcm"
    ),
)
def test_torch_lcm(
    dtype_and_x,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
):
    input_dtype, x = dtype_and_x
    helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="lcm",
        input=np.asarray(x[0], dtype=input_dtype[0]),
        other=np.asarray(x[1], dtype=input_dtype[1]),
    )


# einsum
@handle_cmd_line_args
@given(
    eq_n_op_n_shp=st.sampled_from(
        [
            ("ii", (np.arange(25).reshape(5, 5),), ()),
            ("ii->i", (np.arange(25).reshape(5, 5),), (5,)),
            ("ij,j", (np.arange(25).reshape(5, 5), np.arange(5)), (5,)),
        ]
    ),
    dtype=helpers.get_dtypes("float", full=False),
)
def test_torch_einsum(
    eq_n_op_n_shp,
    dtype,
    as_variable,
    with_out,
    native_array,
):
    eq, operands, _ = eq_n_op_n_shp
    kw = {}
    i = 0
    for x_ in operands:
        kw["x{}".format(i)] = x_
        i += 1
    # len(operands) + 1 because of the equation
    num_positional_args = len(operands) + 1
    helpers.test_frontend_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend="torch",
        fn_tree="einsum",
        equation=eq,
        **kw,
    )

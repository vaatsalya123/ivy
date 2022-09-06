"""Collection of tests for unified linear algebra functions."""


# global
import sys

import numpy as np
from hypothesis import given, strategies as st

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


@st.composite
def dtype_value1_value2_axis(
    draw,
    available_dtypes,
    min_value=None,
    max_value=None,
    allow_inf=True,
    exclude_min=False,
    exclude_max=False,
    min_num_dims=1,
    max_num_dims=10,
    min_dim_size=1,
    max_dim_size=10,
    specific_dim_size=3,
):
    # For cross product, a dim with size 3 is required
    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    axis = draw(helpers.ints(min_value=0, max_value=len(shape)))
    # make sure there is a dim with specific dim size
    shape = list(shape)
    shape = shape[:axis] + [specific_dim_size] + shape[axis:]
    shape = tuple(shape)

    dtype = draw(available_dtypes)

    values = []
    for i in range(2):
        values.append(
            draw(
                helpers.array_values(
                    dtype=dtype,
                    shape=shape,
                    min_value=min_value,
                    max_value=max_value,
                    allow_inf=allow_inf,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                )
            )
        )

    value1, value2 = values[0], values[1]
    return dtype, value1, value2, axis


@st.composite
def _get_dtype_value1_value2_axis_for_tensordot(
    draw,
    available_dtypes,
    min_value=None,
    max_value=None,
    allow_inf=True,
    exclude_min=False,
    exclude_max=False,
    min_num_dims=1,
    max_num_dims=10,
    min_dim_size=1,
    max_dim_size=10,
):

    shape = draw(
        helpers.get_shape(
            allow_none=False,
            min_num_dims=min_num_dims,
            max_num_dims=max_num_dims,
            min_dim_size=min_dim_size,
            max_dim_size=max_dim_size,
        )
    )
    axis = draw(helpers.ints(min_value=1, max_value=len(shape)))
    dtype = draw(available_dtypes)

    values = []
    for i in range(2):
        values.append(
            draw(
                helpers.array_values(
                    dtype=dtype,
                    shape=shape,
                    min_value=min_value,
                    max_value=max_value,
                    allow_inf=allow_inf,
                    exclude_min=exclude_min,
                    exclude_max=exclude_max,
                )
            )
        )

    value1, value2 = values[0], values[1]
    value1 = np.asarray(value1, dtype=dtype)
    value2 = np.asarray(value2, dtype=dtype)
    if not isinstance(axis, list):
        value2 = value2.transpose(
            [k for k in range(len(shape) - axis, len(shape))]
            + [k for k in range(0, len(shape) - axis)]
        )
    return dtype, value1, value2, axis


@st.composite
def _get_dtype_and_matrix(draw, *, symmetric=False):
    # batch_shape, shared, random_size
    input_dtype = draw(st.shared(st.sampled_from(ivy_np.valid_float_dtypes)))
    random_size = draw(helpers.ints(min_value=2, max_value=4))
    batch_shape = draw(helpers.get_shape(min_num_dims=1, max_num_dims=3))
    if symmetric:
        num_independnt_vals = int((random_size**2) / 2 + random_size / 2)
        array_vals_flat = np.array(
            draw(
                helpers.array_values(
                    dtype=input_dtype,
                    shape=tuple(list(batch_shape) + [num_independnt_vals]),
                    min_value=2,
                    max_value=5,
                )
            )
        )
        array_vals = np.zeros(batch_shape + (random_size, random_size))
        c = 0
        for i in range(random_size):
            for j in range(random_size):
                if j < i:
                    continue
                array_vals[..., i, j] = array_vals_flat[..., c]
                array_vals[..., j, i] = array_vals_flat[..., c]
                c += 1
        return input_dtype, array_vals.tolist()
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple(list(batch_shape) + [random_size, random_size]),
            min_value=2,
            max_value=5,
        )
    )


@st.composite
def _get_first_matrix_and_dtype(draw):
    # batch_shape, random_size, shared
    input_dtype = draw(
        st.shared(
            st.sampled_from(draw(helpers.get_dtypes("numeric", full=True))),
            key="shared_dtype",
        )
    )
    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    random_size = draw(helpers.ints(min_value=2, max_value=4))
    batch_shape = draw(
        st.shared(helpers.get_shape(min_num_dims=1, max_num_dims=3), key="shape")
    )
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple(list(batch_shape) + [random_size, shared_size]),
            min_value=2,
            max_value=5,
        )
    )


@st.composite
def _get_second_matrix_and_dtype(draw):
    # batch_shape, shared, random_size
    input_dtype = draw(
        st.shared(
            st.sampled_from(draw(helpers.get_dtypes("numeric", full=True))),
            key="shared_dtype",
        )
    )
    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    random_size = draw(helpers.ints(min_value=2, max_value=4))
    batch_shape = draw(
        st.shared(helpers.get_shape(min_num_dims=1, max_num_dims=3), key="shape")
    )
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple(list(batch_shape) + [shared_size, random_size]),
            min_value=2,
            max_value=5,
        )
    )


# vector_to_skew_symmetric_matrix
@st.composite
def _get_dtype_and_vector(draw):
    # batch_shape, shared, random_size
    input_dtype = draw(st.sampled_from(ivy_np.valid_numeric_dtypes))
    batch_shape = draw(helpers.get_shape(min_num_dims=2, max_num_dims=4))
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple(list(batch_shape) + [3]),
            min_value=2,
            max_value=5,
        )
    )


@handle_cmd_line_args
@given(
    dtype_x=_get_dtype_and_vector(),
)
def test_vector_to_skew_symmetric_matrix(
    *,
    dtype_x,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="vector_to_skew_symmetric_matrix",
        vector=np.asarray(x, dtype=input_dtype),
    )


# matrix_power
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_value=0,
        max_value=50,
        shape=helpers.ints(min_value=2, max_value=8).map(lambda x: tuple([x, x])),
    ),
    n=helpers.ints(min_value=1, max_value=8),
)
def test_matrix_power(
    *,
    dtype_x,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
    n,
):
    dtype, x = dtype_x

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=2,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="matrix_power",
        rtol_=1e-1,
        atol_=1e-1,
        x=np.asarray(x, dtype=dtype),
        n=n,
    )


# matmul
@handle_cmd_line_args
@given(
    x=_get_first_matrix_and_dtype(),
    y=_get_second_matrix_and_dtype(),
)
def test_matmul(
    *,
    x,
    y,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype1, x_1 = x
    input_dtype2, y_1 = y
    input_dtype = [input_dtype1, input_dtype2]
    as_variable = [as_variable, as_variable]
    native_array = [native_array, native_array]
    container = [container, container]

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=2,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="matmul",
        rtol_=1e-2,
        atol_=1e-2,
        x1=np.asarray(x_1, dtype=input_dtype1),
        x2=np.asarray(y_1, dtype=input_dtype2),
    )


# det
@handle_cmd_line_args
@given(
    dtype_x=_get_dtype_and_matrix(),
)
def test_det(
    *,
    dtype_x,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="det",
        x=np.asarray(x, dtype=input_dtype),
    )


# eigh
@handle_cmd_line_args
@given(
    dtype_x=_get_dtype_and_matrix(symmetric=True),
)
def test_eigh(
    *,
    dtype_x,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    x = np.asarray(x, dtype=input_dtype)
    results = helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="eigh",
        x=x,
        test_values=False,
        return_flat_np_arrays=True,
    )
    if results is None:
        return
    ret_np_flat, ret_from_np_flat = results
    eigenvalues_np, eigenvectors_np = ret_np_flat
    reconstructed_np = None
    for eigenvalue, eigenvector in zip(eigenvalues_np, eigenvectors_np):
        if reconstructed_np is not None:
            reconstructed_np += eigenvalue * np.matmul(
                eigenvector.reshape(1, -1), eigenvector.reshape(-1, 1)
            )
        else:
            reconstructed_np = eigenvalue * np.matmul(
                eigenvector.reshape(1, -1), eigenvector.reshape(-1, 1)
            )
    eigenvalues_from_np, eigenvectors_from_np = ret_from_np_flat
    reconstructed_from_np = None
    for eigenvalue, eigenvector in zip(eigenvalues_from_np, eigenvectors_from_np):
        if reconstructed_from_np is not None:
            reconstructed_from_np += eigenvalue * np.matmul(
                eigenvector.reshape(1, -1), eigenvector.reshape(-1, 1)
            )
        else:
            reconstructed_from_np = eigenvalue * np.matmul(
                eigenvector.reshape(1, -1), eigenvector.reshape(-1, 1)
            )
    # value test
    helpers.assert_all_close(
        reconstructed_np, reconstructed_from_np, rtol=1e-1, atol=1e-2
    )


# eigvalsh
@handle_cmd_line_args
@given(
    dtype_x=_get_dtype_and_matrix(symmetric=True),
)
def test_eigvalsh(
    *,
    dtype_x,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="eigvalsh",
        rtol_=1e-3,
        x=np.asarray(x, dtype=input_dtype),
    )


# inv
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_value=0,
        max_value=50,
        shape=helpers.ints(min_value=2, max_value=20).map(lambda x: tuple([x, x])),
    ).filter(lambda x: np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon),
)
def test_inv(
    *,
    dtype_x,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="inv",
        x=np.asarray(x, dtype=input_dtype),
    )


# matrix_transpose
@handle_cmd_line_args
@given(
    dtype_x=_get_first_matrix_and_dtype(),
)
def test_matrix_transpose(
    *,
    dtype_x,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="matrix_transpose",
        x=np.asarray(x, dtype=input_dtype),
    )


# outer
@handle_cmd_line_args
@given(
    dtype_xy=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        num_arrays=2,
        min_value=1,
        max_value=50,
        min_num_dims=1,
        max_num_dims=1,
    ),
)
def test_outer(
    *,
    dtype_xy,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    types, arrays = dtype_xy
    type1, type2 = types
    x1, x2 = arrays
    input_dtype = [type1, type2]
    as_variable = [as_variable, as_variable]
    native_array = [native_array, native_array]
    container = [container, container]

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=2,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="outer",
        x1=np.asarray(x1, input_dtype[0]),
        x2=np.asarray(x2, input_dtype[1]),
    )


# slogdet
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_value=0,
        max_value=50,
        shape=helpers.ints(min_value=2, max_value=20).map(lambda x: tuple([x, x])),
    ),
)
def test_slogdet(
    *,
    dtype_x,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="slogdet",
        x=np.asarray(x, dtype=input_dtype),
    )


# solve
@st.composite
def _get_first_matrix(draw):
    # batch_shape, random_size, shared
    input_dtype = draw(
        st.shared(st.sampled_from(ivy_np.valid_float_dtypes), key="shared_dtype")
    )
    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype,
            shape=tuple([shared_size, shared_size]),
            min_value=2,
            max_value=5,
        ).filter(lambda x: np.linalg.cond(x) < 1 / sys.float_info.epsilon)
    )


@st.composite
def _get_second_matrix(draw):
    # batch_shape, shared, random_size
    input_dtype = draw(
        st.shared(st.sampled_from(ivy_np.valid_float_dtypes), key="shared_dtype")
    )
    shared_size = draw(
        st.shared(helpers.ints(min_value=2, max_value=4), key="shared_size")
    )
    return input_dtype, draw(
        helpers.array_values(
            dtype=input_dtype, shape=tuple([shared_size, 1]), min_value=2, max_value=5
        )
    )


@handle_cmd_line_args
@given(
    x=_get_first_matrix(),
    y=_get_second_matrix(),
)
def test_solve(
    *,
    x,
    y,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype1, x1 = x
    input_dtype2, x2 = y
    input_dtype = [input_dtype1, input_dtype2]
    as_variable = [as_variable, as_variable]
    native_array = [native_array, native_array]
    container = [container, container]

    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=2,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="solve",
        rtol_=1e-2,
        atol_=1e-2,
        x1=np.asarray(x1, dtype=input_dtype1),
        x2=np.asarray(x2, dtype=input_dtype2),
    )


# svdvals
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_value=0,
        max_value=50,
        min_num_dims=2,
    ),
)
def test_svdvals(
    *,
    dtype_x,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    input_dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="svdvals",
        x=np.asarray(x, dtype=input_dtype),
    )


# tensordot
@handle_cmd_line_args
@given(
    dtype_x1_x2_axis=_get_dtype_value1_value2_axis_for_tensordot(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=3,
        max_num_dims=8,
        min_dim_size=1,
        max_dim_size=15,
    ),
)
def test_tensordot(
    *,
    dtype_x1_x2_axis,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):

    (
        dtype,
        x1,
        x2,
        axis,
    ) = dtype_x1_x2_axis

    as_variable = [as_variable, as_variable]
    native_array = [native_array, native_array]
    container = [container, container]

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=2,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="tensordot",
        x1=x1,
        x2=x2,
        axes=axis,
    )


# trace
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=50,
    ),
    offset=helpers.ints(min_value=-10, max_value=10),
)
def test_trace(
    *,
    dtype_x,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
    offset,
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="trace",
        x=np.asarray(x, dtype=dtype),
        offset=offset,
    )


# vecdot
@handle_cmd_line_args
@given(
    dtype_x1_x2_axis=dtype_value1_value2_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
    ),
)
def test_vecdot(
    *,
    dtype_x1_x2_axis,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):

    as_variable = [as_variable, as_variable]
    native_array = [native_array, native_array]
    container = [container, container]
    dtype, x1, x2, axis = dtype_x1_x2_axis
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=2,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="vecdot",
        x1=np.asarray(x1, dtype=dtype),
        x2=np.asarray(x2, dtype=dtype),
        axis=axis,
    )


# vector_norm
@handle_cmd_line_args
@given(
    dtype_values_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=5,
        min_axis=-2,
        max_axis=1,
    ),
    kd=st.booleans(),
    ord=helpers.ints(min_value=1, max_value=2),
)
def test_vector_norm(
    *,
    dtype_values_axis,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
    kd,
    ord,
):
    dtype, x, axis = dtype_values_axis
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="vector_norm",
        rtol_=1e-2,
        atol_=1e-2,
        x=np.asarray(x, dtype=dtype),
        axis=axis,
        keepdims=kd,
        ord=ord,
    )


# pinv
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        large_value_safety_factor=10,
        small_value_safety_factor=1.5,
    ),
    rtol=st.floats(1e-5, 1e-3),
)
def test_pinv(
    *,
    dtype_x,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
    rtol,
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="pinv",
        x=np.asarray(x, dtype=dtype),
        rtol=rtol,
    )


# qr
@handle_cmd_line_args
@given(
    dtype_x=_get_dtype_and_matrix(),
    mode=st.sampled_from(("reduced", "complete")),
)
def test_qr(
    *,
    dtype_x,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
    mode,
):
    dtype, x = dtype_x
    results = helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="qr",
        x=np.asarray(x, dtype=dtype),
        mode=mode,
        test_values=False,
        return_flat_np_arrays=True,
    )
    if results is None:
        return

    ret_np_flat, ret_from_np_flat = results

    q_np_flat, r_np_flat = ret_np_flat
    q_from_np_flat, r_from_np_flat = ret_from_np_flat

    reconstructed_np_flat = np.matmul(q_np_flat, r_np_flat)
    reconstructed_from_np_flat = np.matmul(q_from_np_flat, r_from_np_flat)

    # value test
    helpers.assert_all_close(
        reconstructed_np_flat, reconstructed_from_np_flat, rtol=1e-2, atol=1e-2
    )


# svd
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_num_dims=3,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=5,
        min_value=0.1,
        max_value=10.0,
    ),
    fm=st.booleans(),
)
def test_svd(
    *,
    dtype_x,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
    fm,
):
    dtype, x = dtype_x

    results = helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="svd",
        x=np.asarray(x, dtype=dtype),
        full_matrices=fm,
        test_values=False,
        return_flat_np_arrays=True,
    )
    if results is None:
        return

    # value test based on recreating the original matrix and testing the consistency
    ret_flat_np, ret_from_gt_flat_np = results
    U, S, Vh = ret_flat_np
    m = U.shape[-1]
    n = Vh.shape[-1]
    S = np.expand_dims(S, -2) if m > n else np.expand_dims(S, -1)
    U_gt, S_gt, Vh_gt = ret_from_gt_flat_np
    S_gt = np.expand_dims(S_gt, -2) if m > n else np.expand_dims(S_gt, -1)

    with ivy.functional.backends.numpy.use:
        S_mat = S * ivy.eye(U.shape[-1], Vh.shape[-2], batch_shape=U.shape[:-2]).data
        S_mat_gt = (
            S_gt
            * ivy.eye(U_gt.shape[-1], Vh_gt.shape[-2], batch_shape=U_gt.shape[:-2]).data
        )
    reconstructed = np.matmul(np.matmul(U, S_mat), Vh)
    reconstructed_gt = np.matmul(np.matmul(U_gt, S_mat_gt), Vh_gt)

    # value test
    helpers.assert_all_close(reconstructed, reconstructed_gt, atol=1e-04)
    helpers.assert_all_close(reconstructed, np.asarray(x, dtype=dtype), atol=1e-04)


# matrix_norm
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        min_value=-10,
        max_value=10,
    ),
    kd=st.booleans(),
    ord=helpers.ints(min_value=1, max_value=2) | st.sampled_from(("fro", "nuc")),
)
def test_matrix_norm(
    *,
    dtype_x,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
    kd,
    ord,
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="matrix_norm",
        x=np.asarray(x, dtype=dtype),
        ord=ord,
        keepdims=kd,
    )


# matrix_rank
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", index=1, full=True),
        min_num_dims=2,
        min_value=-1e05,
        max_value=1e05,
    ),
    rtol=st.floats(allow_nan=False, allow_infinity=False) | st.just(None),
)
def test_matrix_rank(
    *,
    dtype_x,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
    rtol,
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="matrix_rank",
        x=np.asarray(x, dtype=dtype),
        rtol=rtol,
    )


# cholesky
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_value=0,
        max_value=10,
        shape=helpers.ints(min_value=2, max_value=5).map(lambda x: tuple([x, x])),
    ).filter(
        lambda x: np.linalg.cond(x[1]) < 1 / sys.float_info.epsilon
        and np.linalg.det(np.asarray(x[1])) != 0
    ),
    upper=st.booleans(),
)
def test_cholesky(
    *,
    dtype_x,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
    upper,
):
    dtype, x = dtype_x
    x = np.asarray(x, dtype=dtype)
    x = (
        np.matmul(x.T, x) + np.identity(x.shape[0]) * 1e-3
    )  # make symmetric positive-definite

    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="cholesky",
        x=x,
        upper=upper,
        rtol_=1e-3,
        atol_=1e-3,
    )


# cross
@handle_cmd_line_args
@given(
    dtype_x1_x2_axis=dtype_value1_value2_axis(
        available_dtypes=helpers.get_dtypes("numeric"),
        min_num_dims=1,
        max_num_dims=10,
        min_dim_size=3,
        max_dim_size=3,
    ),
)
def test_cross(
    *,
    dtype_x1_x2_axis,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
):
    dtype, x1, x2, axis = dtype_x1_x2_axis
    as_variable = [as_variable, as_variable]
    native_array = [native_array, native_array]
    container = [container, container]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=2,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="cross",
        x1=np.asarray(x1, dtype=dtype),
        x2=np.asarray(x2, dtype=dtype),
        axis=axis,
    )


# diagonal
@handle_cmd_line_args
@given(
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric", full=True),
        min_num_dims=2,
        max_num_dims=2,
        min_dim_size=1,
        max_dim_size=50,
    ),
    offset=helpers.ints(min_value=-10, max_value=50),
    axes=st.lists(
        helpers.ints(min_value=-2, max_value=1), min_size=2, max_size=2, unique=True
    ).filter(lambda axes: axes[0] % 2 != axes[1] % 2),
)
def test_diagonal(
    *,
    dtype_x,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
    offset,
    axes,
):
    dtype, x = dtype_x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="diagonal",
        x=np.asarray(x, dtype=dtype),
        offset=offset,
        axis1=axes[0],
        axis2=axes[1],
    )

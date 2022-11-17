# global
from hypothesis import assume, strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
import ivy_tests.test_ivy.test_frontends.test_numpy.helpers as np_frontend_helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test
import ivy


@st.composite
def _get_clip_inputs(draw):
    shape = draw(
        helpers.get_shape(
            min_num_dims=1, max_num_dims=5, min_dim_size=2, max_dim_size=10
        )
    )
    x_dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            shape=shape,
        )
    )
    min = draw(st.booleans())
    if min:
        min = draw(
            helpers.array_values(
                dtype=x_dtype[0], shape=shape, min_value=-50, max_value=5
            )
        )
        max = draw(st.booleans())
        max = (
            draw(
                helpers.array_values(
                    dtype=x_dtype[0], shape=shape, min_value=6, max_value=50
                )
            )
            if max
            else None
        )
    else:
        min = None
        max = draw(
            helpers.array_values(
                dtype=x_dtype[0], shape=shape, min_value=6, max_value=50
            )
        )
    return x_dtype, x, min, max


# clip
@handle_frontend_test(
    fn_tree="numpy.clip",
    input_and_ranges=_get_clip_inputs(),
    where=np_frontend_helpers.where(),
)
def test_numpy_clip(
    input_and_ranges,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x, min, max = input_and_ranges
    dtype, input_dtype, casting = np_frontend_helpers.handle_dtype_and_casting(
        dtypes=input_dtype,
        get_dtypes_kind="numeric",
    )
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
        a_min=min,
        a_max=max,
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# cbrt
@handle_frontend_test(
    fn_tree="numpy.cbrt",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_cbrt(
    dtype_and_x,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    dtype, input_dtype, casting = np_frontend_helpers.handle_dtype_and_casting(
        dtypes=input_dtype,
        get_dtypes_kind="numeric",
    )
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# sqrt
@handle_frontend_test(
    fn_tree="numpy.sqrt",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_sqrt(
    dtype_and_x,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    dtype, input_dtype, casting = np_frontend_helpers.handle_dtype_and_casting(
        dtypes=input_dtype,
        get_dtypes_kind="numeric",
    )
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# square
@handle_frontend_test(
    fn_tree="numpy.square",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric")
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_square(
    dtype_and_x,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    dtype, input_dtype, casting = np_frontend_helpers.handle_dtype_and_casting(
        dtypes=input_dtype,
        get_dtypes_kind="numeric",
    )
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtype,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# absolute
@handle_frontend_test(
    fn_tree="numpy.absolute",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    where=np_frontend_helpers.where(),
)
def test_numpy_absolute(
    dtype_and_x,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    dtype, input_dtype, casting = np_frontend_helpers.handle_dtype_and_casting(
        dtypes=input_dtype,
        get_dtypes_kind="float",
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# fabs
@handle_frontend_test(
    fn_tree="numpy.fabs",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    where=np_frontend_helpers.where(),
)
def test_numpy_fabs(
    dtype_and_x,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    dtype, input_dtype, casting = np_frontend_helpers.handle_dtype_and_casting(
        dtypes=input_dtype,
        get_dtypes_kind="float",
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# heaviside
@handle_frontend_test(
    fn_tree="numpy.heaviside",
    x1_x2=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        shared_dtype=True,
    ),
    where=np_frontend_helpers.where(),
)
def test_numpy_heaviside(
    x1_x2,
    where,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtypes, (x1_list, x2_list) = x1_x2
    dtype, input_dtypes, casting = np_frontend_helpers.handle_dtype_and_casting(
        dtypes=input_dtypes,
        get_dtypes_kind="float",
    )
    where, as_variable, native_array = np_frontend_helpers.handle_where_and_array_bools(
        where=where,
        input_dtype=input_dtypes,
        as_variable=as_variable,
        native_array=native_array,
    )
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x1=x1_list,
        x2=x2_list,
        out=None,
        where=where,
        casting=casting,
        order="K",
        dtype=dtype,
        subok=True,
    )


# nan_to_num
@handle_frontend_test(
    fn_tree="numpy.nan_to_num",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    posinf=st.one_of(st.none(), st.floats()),
    neginf=st.one_of(st.none(), st.floats()),
)
def test_numpy_nan_to_num(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
    posinf,
    neginf,
):
    input_dtype, x = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        nan=0.0,
        posinf=posinf,
        neginf=neginf,
    )


# real_if_close
@handle_frontend_test(
    fn_tree="numpy.real_if_close",
    dtype_and_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_numpy_real_if_close(
    dtype_and_x,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtype, x = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=x[0],
    )


# interp
@handle_frontend_test(
    fn_tree="numpy.interp",
    xp_and_fp=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        num_arrays=2,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=3,
        min_value=-10000,
        max_value=10000,
    ),
    x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
    left=st.one_of(st.none(), st.floats()),
    right=st.one_of(st.none(), st.floats()),
    period=st.one_of(
        st.none(),
        st.floats(
            allow_nan=False,
            allow_infinity=False,
            allow_subnormal=False,
            min_value=0.1,
            max_value=1.0e5,
            exclude_min=True,
        ),
    ),
)
def test_numpy_interp(
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
    xp_and_fp,
    x,
    left,
    right,
    period,
):
    input_dtypes, xp_fp = xp_and_fp
    xp = ivy.array(xp_fp[0])
    fp = ivy.array(xp_fp[1])
    if period is None:
        xp_order = ivy.argsort(xp)
        xp = xp[xp_order]
        fp = fp[xp_order]
        previous = xp[0]
        for i in xp[1:]:
            assume(i > previous)
            previous = i
    x_dtype, x = x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes + x_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        x=x[0],
        xp=xp,
        fp=fp,
        left=left,
        right=right,
        period=period,
    )


@handle_frontend_test(
    fn_tree="numpy.convolve",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float", full=True),
        min_num_dims=1,
        max_num_dims=1,
        num_arrays=2,
        shared_dtype=True,
    ),
    mode=st.sampled_from(["valid", "same", "full"]),
)
def test_numpy_convolve(
    dtype_and_x,
    mode,
    as_variable,
    num_positional_args,
    native_array,
    frontend,
    fn_tree,
    on_device,
):
    input_dtypes, xs = dtype_and_x
    np_frontend_helpers.test_frontend_function(
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        a=xs[0],
        v=xs[1],
        mode=mode,
    )

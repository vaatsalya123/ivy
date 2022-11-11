"""Collection of tests for unified neural network layers."""

# global
from hypothesis import given, strategies as st, assume

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

# Linear #
# -------#


@st.composite
def x_and_linear(draw, dtypes):
    dtype = draw(dtypes)
    in_features = draw(helpers.ints(min_value=1, max_value=2))
    out_features = draw(helpers.ints(min_value=1, max_value=2))

    x_shape = (
        1,
        1,
        in_features,
    )
    weight_shape = (1,) + (out_features,) + (in_features,)
    bias_shape = (
        1,
        out_features,
    )

    x = draw(
        helpers.array_values(dtype=dtype[0], shape=x_shape, min_value=0, max_value=1)
    )
    weight = draw(
        helpers.array_values(
            dtype=dtype[0], shape=weight_shape, min_value=0, max_value=1
        )
    )
    bias = draw(
        helpers.array_values(dtype=dtype[0], shape=bias_shape, min_value=0, max_value=1)
    )
    return dtype, x, weight, bias


# linear
@handle_cmd_line_args
@given(
    dtype_x_weight_bias=x_and_linear(
        dtypes=helpers.get_dtypes("float", full=False),
    ),
    num_positional_args=helpers.num_positional_args(fn_name="linear"),
)
def test_linear(
    *,
    dtype_x_weight_bias,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, weight, bias = dtype_x_weight_bias
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="linear",
        ground_truth_backend="jax",
        rtol_=1e-02,
        atol_=1e-02,
        test_gradients=True,
        x=x,
        weight=weight,
        bias=bias,
    )


# Dropout #
# --------#

# dropout
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    prob=helpers.floats(min_value=0, max_value=0.9),
    scale=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="dropout"),
)
def test_dropout(
    *,
    dtype_and_x,
    prob,
    scale,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x = dtype_and_x
    ret = helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="dropout",
        test_values=False,
        x=x[0],
        prob=prob,
        scale=scale,
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    for u in ret:
        # cardinality test
        assert u.shape == x[0].shape


# dropout1d
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=0,
        max_value=50,
        allow_inf=False,
        min_num_dims=2,
        max_num_dims=3,
        min_dim_size=1,
        max_dim_size=5,
    ),
    prob=helpers.floats(min_value=0, max_value=0.9),
    training=st.booleans(),
    data_format=st.sampled_from(["NWC", "NCW"]),
    num_positional_args=helpers.num_positional_args(fn_name="dropout1d"),
)
def test_dropout1d(
    *,
    dtype_and_x,
    prob,
    training,
    data_format,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x = dtype_and_x
    ret, gt_ret = helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="dropout1d",
        test_values=False,
        x=x[0],
        prob=prob,
        training=training,
        data_format=data_format,
        return_flat_np_arrays=True,
    )
    ret = helpers.flatten_and_to_np(ret=ret)
    gt_ret = helpers.flatten_and_to_np(ret=gt_ret)
    for u, v, w in zip(ret, gt_ret, x):
        # cardinality test
        assert u.shape == v.shape == w.shape


# Attention #
# ----------#


@st.composite
def x_and_scaled_attention(draw, dtypes):
    dtype = draw(dtypes)
    num_queries = draw(helpers.ints(min_value=1, max_value=2))
    num_keys = draw(helpers.ints(min_value=1, max_value=2))
    feat_dim = draw(helpers.ints(min_value=1, max_value=2))
    scale = draw(helpers.floats(min_value=0.1, max_value=1))

    q_shape = (1,) + (num_queries,) + (feat_dim,)
    k_shape = (1,) + (num_keys,) + (feat_dim,)
    v_shape = (1,) + (num_keys,) + (feat_dim,)
    mask_shape = (1,) + (num_queries,) + (num_keys,)

    q = draw(
        helpers.array_values(dtype=dtype[0], shape=q_shape, min_value=0, max_value=1)
    )
    k = draw(
        helpers.array_values(dtype=dtype[0], shape=k_shape, min_value=0, max_value=1)
    )
    v = draw(
        helpers.array_values(dtype=dtype[0], shape=v_shape, min_value=0, max_value=1)
    )
    mask = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=mask_shape,
            min_value=0,
            max_value=1,
            large_abs_safety_factor=2,
            safety_factor_scale="linear",
        )
    )
    return dtype, q, k, v, mask, scale


# scaled_dot_product_attention
@handle_cmd_line_args
@given(
    dtype_q_k_v_mask_scale=x_and_scaled_attention(
        dtypes=helpers.get_dtypes("float", full=False),
    ),
    num_positional_args=helpers.num_positional_args(
        fn_name="scaled_dot_product_attention"
    ),
)
def test_scaled_dot_product_attention(
    *,
    dtype_q_k_v_mask_scale,
    as_variable,
    num_positional_args,
    with_out,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, q, k, v, mask, scale = dtype_q_k_v_mask_scale
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="scaled_dot_product_attention",
        ground_truth_backend="jax",
        rtol_=1e-02,
        atol_=1e-02,
        test_gradients=True,
        q=q,
        k=k,
        v=v,
        scale=scale,
        mask=mask,
    )


@st.composite
def x_and_mha(draw, dtypes):
    dtype = draw(dtypes)
    num_queries = draw(helpers.ints(min_value=1, max_value=3))
    feat_dim = draw(helpers.ints(min_value=1, max_value=3))
    num_heads = draw(helpers.ints(min_value=1, max_value=3))
    num_keys = draw(helpers.ints(min_value=1, max_value=3))

    x_mha_shape = (num_queries,) + (feat_dim * num_heads,)
    context_shape = (num_keys,) + (2 * feat_dim * num_heads,)
    mask_shape = (num_queries,) + (num_keys,)
    scale = draw(helpers.floats(min_value=0.1, max_value=1))
    x_mha = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=x_mha_shape,
            min_value=0.0999755859375,
            max_value=1,
        )
    )
    context = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=context_shape,
            min_value=0.0999755859375,
            max_value=1,
        )
    )
    mask = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=mask_shape,
            min_value=0.0999755859375,
            max_value=1,
        )
    )
    return dtype, x_mha, scale, num_heads, context, mask


# multi_head_attention
@handle_cmd_line_args
@given(
    dtype_mha=x_and_mha(
        dtypes=helpers.get_dtypes("float", full=False),
    ),
    num_positional_args=helpers.num_positional_args(fn_name="multi_head_attention"),
)
def test_multi_head_attention(
    *,
    dtype_mha,
    as_variable,
    with_out,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x_mha, scale, num_heads, context, mask = dtype_mha
    to_q_fn = lambda x_, v: x_
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="multi_head_attention",
        ground_truth_backend="jax",
        atol_=1e-02,
        rtol_=1e-02,
        test_gradients=True,
        x=x_mha,
        scale=scale,
        num_heads=num_heads,
        context=context,
        mask=mask,
        to_q_fn=to_q_fn,
        to_kv_fn=to_q_fn,
        to_out_fn=to_q_fn,
        to_q_v=None,
        to_kv_v=None,
        to_out_v=None,
    )


# Convolutions #
# -------------#


def _deconv_length(dim_size, stride_size, kernel_size, padding, dilation=1):
    # Get the dilated kernel size
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)

    if padding == "VALID":
        dim_size = dim_size * stride_size + max(kernel_size - stride_size, 0)
    elif padding == "SAME":
        dim_size = dim_size * stride_size

    return dim_size


@st.composite
def x_and_filters(
    draw, dim: int = 2, transpose: bool = False, depthwise=False, general=False
):
    if not isinstance(dim, int):
        dim = draw(dim)
    strides = draw(st.integers(min_value=1, max_value=2))
    padding = draw(st.sampled_from(["SAME", "VALID"]))
    batch_size = 1
    filter_shape = draw(
        helpers.get_shape(
            min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=5
        )
    )
    dtype = draw(helpers.get_dtypes("float", full=False))
    input_channels = draw(st.integers(1, 3))
    output_channels = draw(st.integers(1, 3))
    group_list = [i for i in range(1, 6)]
    if not transpose:
        group_list = list(filter(lambda x: (input_channels % x == 0), group_list))
    else:
        group_list = list(filter(lambda x: (output_channels % x == 0), group_list))
    fc = draw(st.sampled_from(group_list)) if general else 1
    # tensorflow backprop doesn't support dilations more than 1 on CPU
    dilations = 1
    if dim == 2:
        data_format = draw(st.sampled_from(["NCHW"]))
    elif dim == 1:
        data_format = draw(st.sampled_from(["NWC", "NCW"]))
    else:
        data_format = draw(st.sampled_from(["NDHWC", "NCDHW"]))

    x_dim = []
    if transpose:
        output_shape = []
        x_dim = draw(
            helpers.get_shape(
                min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=5
            )
        )
        for i in range(dim):
            output_shape.append(
                ivy.deconv_length(
                    x_dim[i], strides, filter_shape[i], padding, dilations
                )
            )
    else:
        for i in range(dim):
            min_x = filter_shape[i] + (filter_shape[i] - 1) * (dilations - 1)
            x_dim.append(draw(st.integers(min_x, min_x + 1)))
        x_dim = tuple(x_dim)
    if not depthwise:
        if not transpose:
            filter_shape = filter_shape + (input_channels // fc, output_channels * fc)
        else:
            input_channels = input_channels * fc
            filter_shape = filter_shape + (input_channels, output_channels // fc)
    else:
        filter_shape = filter_shape + (input_channels,)
    channel_first = True
    if data_format == "NHWC" or data_format == "NWC" or data_format == "NDHWC":
        x_shape = (batch_size,) + x_dim + (input_channels,)
        channel_first = False
    else:
        x_shape = (batch_size, input_channels) + x_dim
    vals = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=x_shape,
            min_value=0.0,
            max_value=1.0,
        )
    )
    filters = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=filter_shape,
            min_value=0.0,
            max_value=1.0,
        )
    )
    if transpose:
        if general:
            data_format = "channel_first" if channel_first else "channel_last"
        return (
            dtype,
            vals,
            filters,
            dilations,
            data_format,
            strides,
            padding,
            output_shape,
            fc,
        )
    if general:
        data_format = "channel_first" if channel_first else "channel_last"
        return dtype, vals, filters, dilations, data_format, strides, padding, fc

    return dtype, vals, filters, dilations, data_format, strides, padding


# conv1d
@handle_cmd_line_args
@given(
    x_f_d_df=x_and_filters(dim=1),
    num_positional_args=helpers.num_positional_args(fn_name="conv1d"),
)
def test_conv1d(
    *,
    x_f_d_df,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv1d",
        rtol_=1e-02,
        atol_=1e-02,
        test_gradients=True,
        ground_truth_backend="jax",
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


# conv1d_transpose
@handle_cmd_line_args
@given(
    x_f_d_df=x_and_filters(dim=1, transpose=True),
    num_positional_args=helpers.num_positional_args(fn_name="conv1d_transpose"),
)
def test_conv1d_transpose(
    *,
    x_f_d_df,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad, output_shape, fc = x_f_d_df
    assume(not (fw == "tensorflow" and device == "cpu" and dilations > 1))
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv1d_transpose",
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=True,
        # tensorflow does not work with dilations > 1 on cpu
        ground_truth_backend="jax",
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


# conv2d
@handle_cmd_line_args
@given(
    x_f_d_df=x_and_filters(dim=2),
    num_positional_args=helpers.num_positional_args(fn_name="conv2d"),
)
def test_conv2d(
    *,
    x_f_d_df,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv2d",
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=True,
        ground_truth_backend="jax",
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


# conv2d_transpose
@handle_cmd_line_args
@given(
    x_f_d_df=x_and_filters(
        dim=2,
        transpose=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="conv2d_transpose"),
)
def test_conv2d_transpose(
    *,
    x_f_d_df,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad, output_shape, fc = x_f_d_df
    assume(not (fw == "tensorflow" and device == "cpu" and dilations > 1))
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv2d_transpose",
        rtol_=1e-2,
        atol_=1e-2,
        device_=device,
        test_gradients=True,
        # tensorflow does not work with dilations > 1 on cpu
        ground_truth_backend="jax",
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


# depthwise_conv2d
@handle_cmd_line_args
@given(
    x_f_d_df=x_and_filters(
        dim=2,
        depthwise=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="depthwise_conv2d"),
)
def test_depthwise_conv2d(
    *,
    x_f_d_df,
    with_out,
    num_positional_args,
    as_variable,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    assume(not (fw == "tensorflow" and dilations > 1 and stride > 1))
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="depthwise_conv2d",
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=True,
        # tensorflow does not support dilations > 1 and stride > 1
        ground_truth_backend="jax",
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


# conv3d
@handle_cmd_line_args
@given(
    x_f_d_df=x_and_filters(dim=3),
    num_positional_args=helpers.num_positional_args(fn_name="conv3d"),
)
def test_conv3d(
    *,
    x_f_d_df,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad = x_f_d_df
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv3d",
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=True,
        ground_truth_backend="jax",
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        data_format=data_format,
        dilations=dilations,
    )


@handle_cmd_line_args
@given(
    dims=st.shared(st.integers(1, 3), key="dims"),
    x_f_d_df=x_and_filters(dim=st.shared(st.integers(1, 3), key="dims"), general=True),
    x_dilations=st.integers(1, 3),
    num_positional_args=helpers.num_positional_args(fn_name="conv_general_dilated"),
)
def test_conv_general_dilated(
    *,
    dims,
    x_f_d_df,
    x_dilations,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad, fc = x_f_d_df
    assume(not (fw == "tensorflow" and device == "cpu" and dilations > 1))
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv_general_dilated",
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=True,
        # tensorflow does not work with dilations > 1 on cpu
        ground_truth_backend="jax",
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        dims=dims,
        data_format=data_format,
        feature_group_count=fc,
        x_dilations=x_dilations,
        dilations=dilations,
    )


@handle_cmd_line_args
@given(
    dims=st.shared(st.integers(1, 3), key="dims"),
    x_f_d_df=x_and_filters(
        dim=st.shared(st.integers(1, 3), key="dims"), general=True, transpose=True
    ),
    num_positional_args=helpers.num_positional_args(fn_name="conv_general_transpose"),
)
def test_conv_general_transpose(
    *,
    dims,
    x_f_d_df,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad, output_shape, fc = x_f_d_df
    assume(not (fw == "tensorflow" and device == "cpu" and dilations > 1))
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv_general_transpose",
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=True,
        # tensorflow does not work with dilations > 1 on cpu
        ground_truth_backend="jax",
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        dims=dims,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
        feature_group_count=fc,
    )


# conv3d_transpose
@handle_cmd_line_args
@given(
    x_f_d_df=x_and_filters(
        dim=3,
        transpose=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="conv3d_transpose"),
)
def test_conv3d_transpose(
    *,
    x_f_d_df,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, filters, dilations, data_format, stride, pad, output_shape, fc = x_f_d_df
    assume(not (fw == "tensorflow" and device == "cpu" and dilations > 1))
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="conv3d_transpose",
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=True,
        ground_truth_backend="jax",
        x=x,
        filters=filters,
        strides=stride,
        padding=pad,
        output_shape=output_shape,
        data_format=data_format,
        dilations=dilations,
    )


# LSTM #
# -----#


@st.composite
def x_and_lstm(draw, dtypes):
    dtype = draw(dtypes)
    batch_shape = (1,)

    t = draw(helpers.ints(min_value=1, max_value=2))
    _in_ = draw(helpers.ints(min_value=1, max_value=2))
    _out_ = draw(helpers.ints(min_value=1, max_value=2))

    x_lstm_shape = batch_shape + (t,) + (_in_,)
    init_h_shape = batch_shape + (_out_,)
    init_c_shape = init_h_shape
    kernel_shape = (_in_,) + (4 * _out_,)
    recurrent_kernel_shape = (_out_,) + (4 * _out_,)
    bias_shape = (4 * _out_,)
    recurrent_bias_shape = bias_shape

    x_lstm = draw(
        helpers.array_values(
            dtype=dtype[0], shape=x_lstm_shape, min_value=0, max_value=1
        )
    )
    init_h = draw(
        helpers.array_values(
            dtype=dtype[0], shape=init_h_shape, min_value=0, max_value=1
        )
    )
    init_c = draw(
        helpers.array_values(
            dtype=dtype[0], shape=init_c_shape, min_value=0, max_value=1
        )
    )
    kernel = draw(
        helpers.array_values(
            dtype=dtype[0], shape=kernel_shape, min_value=0, max_value=1
        )
    )
    recurrent_kernel = draw(
        helpers.array_values(
            dtype=dtype[0], shape=recurrent_kernel_shape, min_value=0, max_value=1
        )
    )
    lstm_bias = draw(
        helpers.array_values(dtype=dtype[0], shape=bias_shape, min_value=0, max_value=1)
    )
    recurrent_bias = draw(
        helpers.array_values(
            dtype=dtype[0], shape=recurrent_bias_shape, min_value=0, max_value=1
        )
    )
    return (
        dtype,
        x_lstm,
        init_h,
        init_c,
        kernel,
        recurrent_kernel,
        lstm_bias,
        recurrent_bias,
    )


# lstm
@handle_cmd_line_args
@given(
    dtype_lstm=x_and_lstm(
        dtypes=helpers.get_dtypes("float", full=False),
    ),
    num_positional_args=helpers.num_positional_args(fn_name="lstm_update"),
)
def test_lstm_update(
    *,
    dtype_lstm,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    (
        dtype,
        x_lstm,
        init_h,
        init_c,
        kernel,
        recurrent_kernel,
        bias,
        recurrent_bias,
    ) = dtype_lstm
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="lstm_update",
        rtol_=1e-01,
        atol_=1e-01,
        test_gradients=True,
        x=x_lstm,
        init_h=init_h,
        init_c=init_c,
        kernel=kernel,
        recurrent_kernel=recurrent_kernel,
        bias=bias,
        recurrent_bias=recurrent_bias,
    )


@st.composite
def x_and_fft(draw, dtypes):
    min_fft_points = 2
    dtype = draw(dtypes)
    x_dim = draw(
        helpers.get_shape(
            min_dim_size=2, max_dim_size=100, min_num_dims=1, max_num_dims=4
        )
    )
    x = draw(
        helpers.array_values(
            dtype=dtype[0],
            shape=tuple(x_dim),
        )
    )
    dim = draw(
        helpers.get_axis(shape=x_dim, allow_neg=True, allow_none=False, max_size=1)
    )
    norm = draw(st.sampled_from(["backward", "forward", "ortho"]))
    n = draw(st.integers(min_fft_points, 256))
    return dtype, x, dim, norm, n


@handle_cmd_line_args
@given(
    d_x_d_n_n=x_and_fft(helpers.get_dtypes("complex")),
    num_positional_args=helpers.num_positional_args(fn_name="fft"),
)
def test_fft(
    *,
    d_x_d_n_n,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    dtype, x, dim, norm, n = d_x_d_n_n
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="fft",
        rtol_=1e-2,
        atol_=1e-2,
        test_gradients=False,
        ground_truth_backend="numpy",
        x=x,
        dim=dim,
        norm=norm,
        n=n,
    )

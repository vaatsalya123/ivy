"""Collection of tests for unified general functions."""

# global
import time
import jax.numpy as jnp
import pytest
from hypothesis import given, assume, strategies as st
import numpy as np
from collections.abc import Sequence
import torch.multiprocessing as multiprocessing

# local
import threading
import ivy
import ivy.functional.backends.jax
import ivy.functional.backends.tensorflow
import ivy.functional.backends.torch
import ivy.functional.backends.mxnet
import ivy_tests.test_ivy.helpers as helpers
import ivy.functional.backends.numpy as ivy_np
from ivy_tests.test_ivy.helpers import handle_cmd_line_args

# Helpers #
# --------#


def _get_shape_of_list(lst, shape=()):
    if not lst:
        return []
    if not isinstance(lst, Sequence):
        return shape
    if isinstance(lst[0], Sequence):
        length = len(lst[0])
        if not all(len(item) == length for item in lst):
            msg = "not all lists have the same length"
            raise ValueError(msg)
    shape += (len(lst),)
    shape = _get_shape_of_list(lst[0], shape)
    return shape


# Tests #
# ------#

# set_framework
@handle_cmd_line_args
@given(fw_str=st.sampled_from(["numpy", "jax", "torch", "mxnet"]))
def test_set_framework(fw_str, device):
    ivy.set_backend(fw_str)
    ivy.unset_backend()


# use_framework
@handle_cmd_line_args
def test_use_within_use_framework(device):
    with ivy.functional.backends.numpy.use:
        pass
    with ivy.functional.backends.jax.use:
        pass
    with ivy.functional.backends.tensorflow.use:
        pass
    with ivy.functional.backends.torch.use:
        pass
    # with ivy.functional.backends.mxnet.use:
    #     pass


@handle_cmd_line_args
@given(allow_duplicates=st.booleans())
def test_match_kwargs(allow_duplicates):
    def func_a(a, b, c=2):
        pass

    def func_b(a, d, e=5):
        return None

    class ClassA:
        def __init__(self, c, f, g=3):
            pass

    kwargs = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6}
    kwfa, kwfb, kwca = ivy.match_kwargs(
        kwargs, func_a, func_b, ClassA, allow_duplicates=allow_duplicates
    )
    if allow_duplicates:
        assert kwfa == {"a": 0, "b": 1, "c": 2}
        assert kwfb == {"a": 0, "d": 3, "e": 4}
        assert kwca == {"c": 2, "f": 5, "g": 6}
    else:
        assert kwfa == {"a": 0, "b": 1, "c": 2}
        assert kwfb == {"d": 3, "e": 4}
        assert kwca == {"f": 5, "g": 6}


@handle_cmd_line_args
def test_get_referrers_recursive(device):
    class SomeClass:
        def __init__(self):
            self.x = [1, 2]
            self.y = [self.x]

    some_obj = SomeClass()
    refs = ivy.get_referrers_recursive(some_obj.x)
    ref_keys = refs.keys()
    assert len(ref_keys) == 3
    assert "repr" in ref_keys
    assert refs["repr"] == "[1,2]"
    y_id = str(id(some_obj.y))
    y_refs = refs[y_id]
    assert y_refs["repr"] == "[[1,2]]"
    some_obj_dict_id = str(id(some_obj.__dict__))
    assert y_refs[some_obj_dict_id] == "tracked"
    dict_refs = refs[some_obj_dict_id]
    assert dict_refs["repr"] == "{'x':[1,2],'y':[[1,2]]}"
    some_obj_id = str(id(some_obj))
    some_obj_refs = dict_refs[some_obj_id]
    assert some_obj_refs["repr"] == str(some_obj).replace(" ", "")
    assert len(some_obj_refs) == 1


# copy array
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True)
    )
)
def test_copy_array(dtype_and_x, device, fw):
    dtype, x = dtype_and_x
    assume(not (fw == "torch" and dtype in ["uint16", "uint32", "uint64"]))

    # mxnet does not support int16
    assume(not (fw == "mxnet" and dtype == "int16"))

    # smoke test
    x = ivy.array(x, dtype=dtype, device=device)
    ret = ivy.copy_array(x)
    # type test
    assert ivy.is_ivy_array(ret)
    # cardinality test
    assert ret.shape == x.shape
    # value test
    helpers.assert_all_close(ivy.to_numpy(ret), ivy.to_numpy(x))
    assert id(x) != id(ret)


# array_equal
@handle_cmd_line_args
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True), num_arrays=2
    )
)
def test_array_equal(x0_n_x1_n_res, device, fw):
    dtype0, x0 = x0_n_x1_n_res[0][0], x0_n_x1_n_res[1][0]
    dtype1, x1 = x0_n_x1_n_res[0][1], x0_n_x1_n_res[1][1]

    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in (dtype0, dtype1)))
    # torch does not support those dtypes
    assume(
        not (
            fw == "torch"
            and (
                dtype0 in ["uint16", "uint32", "uint64"]
                or dtype1 in ["uint16", "uint32", "uint64"]
            )
        )
    )

    # mxnet does not support int16, and does not support
    # bool for broadcast_equal method used
    assume(
        not (
            fw == "mxnet"
            and (dtype0 in ["int16", "bool"] or dtype1 in ["int16", "bool"])
        )
    )

    # smoke test
    x0 = ivy.array(x0, dtype=dtype0, device=device)
    x1 = ivy.array(x1, dtype=dtype1, device=device)
    res = ivy.array_equal(x0, x1)
    # type test
    assert ivy.is_ivy_array(x0)
    assert ivy.is_ivy_array(x1)
    assert isinstance(res, bool) or ivy.is_ivy_array(res)
    # value test
    assert res == np.array_equal(np.array(x0, dtype=dtype0), np.array(x1, dtype=dtype1))


# arrays_equal
@handle_cmd_line_args
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True), num_arrays=3
    )
)
def test_arrays_equal(x0_n_x1_n_res, device, fw):
    dtype0, x0 = x0_n_x1_n_res[0][0], x0_n_x1_n_res[1][0]
    dtype1, x1 = x0_n_x1_n_res[0][1], x0_n_x1_n_res[1][1]
    dtype2, x2 = x0_n_x1_n_res[0][2], x0_n_x1_n_res[1][2]
    assume(
        not (
            fw == "torch"
            and (
                dtype0 in ["uint16", "uint32", "uint64"]
                or dtype1 in ["uint16", "uint32", "uint64"]
                or dtype2 in ["uint16", "uint32", "uint64"]
            )
        )
    )
    # torch does not support those dtypes
    assume(
        not (
            fw == "mxnet"
            and (dtype0 in ["int16", "bool"] or dtype1 in ["int16", "bool"])
        )
    )
    # mxnet does not support int16, and does not support bool
    # for broadcast_equal method used
    # smoke test
    x0 = ivy.array(x0, dtype=dtype0, device=device)
    x1 = ivy.array(x1, dtype=dtype1, device=device)
    x2 = ivy.array(x2, dtype=dtype2, device=device)
    res = ivy.arrays_equal([x0, x1, x2])
    # type test
    assert ivy.is_ivy_array(x0)
    assert ivy.is_ivy_array(x1)
    assert ivy.is_ivy_array(x2)
    assert isinstance(res, bool) or ivy.is_ivy_array(res)
    # value test
    true_res = (
        np.array_equal(ivy.to_numpy(x0), ivy.to_numpy(x1))
        and np.array_equal(ivy.to_numpy(x0), ivy.to_numpy(x2))
        and np.array_equal(ivy.to_numpy(x1), ivy.to_numpy(x2))
    )
    assert res == true_res


# to_numpy
@handle_cmd_line_args
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True)
    )
)
def test_to_numpy(x0_n_x1_n_res, device, fw):
    dtype, object_in = x0_n_x1_n_res
    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in dtype))
    # smoke test
    ret = ivy.to_numpy(ivy.array(object_in, dtype=dtype, device=device))
    # type test
    assert isinstance(ret, np.ndarray)
    # cardinality test
    assert ret.shape == np.array(object_in).shape
    # value test
    helpers.assert_all_close(ret, np.array(object_in).astype(dtype))


# to_scalar
@handle_cmd_line_args
@given(
    object_in=st.sampled_from([[0.0], [[[1]]], [True], [[1.0]]]),
    dtype=helpers.get_dtypes("valid", full=False),
)
def test_to_scalar(object_in, dtype, device, fw):
    assume(not ("bfloat16" in dtype))
    # bfloat16 is not supported by numpy
    assume(not (fw == "torch" and (dtype in ["uint16", "uint32", "uint64"])))
    # torch does not support those dtypes
    assume(not (fw == "mxnet" and dtype == "int16"))
    # mxnet does not support int16
    # smoke test
    ret = ivy.to_scalar(ivy.array(object_in, dtype=dtype, device=device))
    true_val = ivy.to_numpy(ivy.array(object_in, dtype=dtype)).item()
    # type test
    assert isinstance(ret, type(true_val))
    # value test
    assert ivy.to_scalar(ivy.array(object_in, dtype=dtype, device=device)) == true_val


# to_list
@handle_cmd_line_args
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True)
    )
)
def test_to_list(x0_n_x1_n_res, device, fw):
    dtype, object_in = x0_n_x1_n_res
    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in dtype))
    assume(dtype in ivy.valid_dtypes)
    # smoke test
    arr = ivy.array(object_in, dtype=dtype, device=device)
    ret = ivy.to_list(arr)
    # type test (result won't be a list if input is 0 dimensional
    if arr.ndim != 0:
        assert isinstance(ret, list)
    # cardinality test
    assert _get_shape_of_list(ret) == _get_shape_of_list(object_in)
    # value test
    assert np.allclose(
        np.nan_to_num(
            np.asarray(ivy.to_list(ivy.array(object_in, dtype=dtype, device=device))),
            posinf=np.inf,
            neginf=-np.inf,
        ),
        np.nan_to_num(np.array(object_in).astype(dtype), posinf=np.inf, neginf=-np.inf),
    )


# shape
@handle_cmd_line_args
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True)
    ),
    as_array=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="shape"),
)
def test_shape(
    x0_n_x1_n_res,
    as_array,
    as_variable,
    num_positional_args,
    native_array,
    container,
    device,
    fw,
):
    dtype, x = x0_n_x1_n_res
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=False,
        fw=fw,
        fn_name="shape",
        x=np.asarray(x, dtype=dtype),
        as_array=as_array,
    )


# get_num_dims
@handle_cmd_line_args
@given(
    x0_n_x1_n_res=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=True)
    ),
    as_tensor=st.booleans(),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
)
def test_get_num_dims(x0_n_x1_n_res, as_tensor, tensor_fn, device, fw):
    dtype, object_in = x0_n_x1_n_res
    assume(
        not (
            fw == "torch"
            and (
                dtype in ["uint16", "uint32", "uint64"]
                or (
                    dtype not in ivy_np.valid_float_dtypes
                    and tensor_fn == helpers.var_fn
                )
            )
        )
    )
    # torch does not support those dtypes
    ret = ivy.get_num_dims(tensor_fn(object_in, dtype=dtype, device=device), as_tensor)
    # type test
    if as_tensor:
        assert ivy.is_ivy_array(ret)
    else:
        assert isinstance(ret, int)
        ret = ivy.array(ret)
    # cardinality test
    assert list(ret.shape) == []
    # value test
    assert np.array_equal(
        ivy.to_numpy(ret), np.asarray(len(np.asarray(object_in).shape), np.int32)
    )


# clip_vector_norm
@handle_cmd_line_args
@given(
    x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        large_value_safety_factor=20,
        small_value_safety_factor=2.5,
    ),
    max_norm=st.floats(),
    p=st.floats(),
    as_variable=st.booleans(),
    with_out=st.booleans(),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_clip_vector_norm(
    x,
    max_norm,
    p,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    dtype, x = x[0], x[1]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=2,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="clip_vector_norm",
        x=np.asarray(x, dtype=dtype),
        max_norm=max_norm,
        p=p,
    )


# unstack
@handle_cmd_line_args
@given(
    x_n_dtype_axis=helpers.dtype_values_axis(
        available_dtypes=helpers.get_dtypes("valid"),
        min_num_dims=5,
        min_axis=1,
        max_axis=4,
    ),
    keepdims=st.booleans(),
)
def test_unstack(
    x_n_dtype_axis,
    keepdims,
    as_variable,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    # smoke test
    dtype, x, axis = x_n_dtype_axis
    if axis >= len(np.asarray(x, dtype=dtype).shape):
        axis = len(np.asarray(x, dtype=dtype).shape) - 1
    if fw == "torch" and dtype in ["uint16", "uint32", "uint64"]:
        return
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=2,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="unstack",
        x=np.asarray(x, dtype=dtype),
        axis=axis,
        keepdims=keepdims,
    )


# fourier_encode
# @given(
#     x=helpers.dtype_and_values(ivy_np.valid_float_dtypes, min_num_dims=1),
#     max_freq=helpers.dtype_and_values(ivy_np.valid_float_dtypes),
#     num_bands=st.integers(min_value=1,max_value=100000),
#     as_variable=st.booleans(),
#     num_positional_args=st.integers(0, 3),
#     native_array=st.booleans(),
#     container=st.booleans(),
#     instance_method=st.booleans(),
# )
# def test_fourier_encode(
#     x,
#     max_freq,
#     num_bands,
#     as_variable,
#     num_positional_args,
#     native_array,
#     container,
#     instance_method,
#     device,
#     call,
#     fw
# ):
#     # smoke test
#     dtype_x, x = x
#     dtype_max_freq, max_freq = max_freq
#     if fw == "torch" and dtype_x in ["uint16", "uint32", "uint64"]:
#         return
#     helpers.test_function(
#         dtype_x,
#         as_variable,
#         False,
#         num_positional_args,
#         native_array,
#         container,
#         instance_method,
#         fw,
#         "fourier_encode",
#         x=np.asarray(x, dtype=dtype_x),
#         max_freq=np.asarray(max_freq,dtype=dtype_max_freq),
#         num_bands=num_bands
#     )


# indices_where
@given(
    x=helpers.dtype_and_values(available_dtypes=(ivy_np.bool,)),
    with_out=st.booleans(),
    as_variable=st.booleans(),
    native_array=st.booleans(),
    container=st.booleans(),
    instance_method=st.booleans(),
)
def test_indices_where(
    x,
    with_out,
    as_variable,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    dtype, x = x
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=1,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="indices_where",
        x=np.asarray(x, dtype=dtype),
    )


@st.composite
def _dtype_indices_depth(draw):
    depth = draw(helpers.ints(min_value=2, max_value=100))
    dtype_and_indices = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("numeric"),
            min_value=0,
            max_value=depth - 1,
            small_value_safety_factor=2.5,
        )
    )
    return dtype_and_indices, depth


# one_hot
@handle_cmd_line_args
@given(
    dtype_indices_depth=_dtype_indices_depth(),
)
def test_one_hot(
    dtype_indices_depth,
    with_out,
    as_variable,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    dtype_and_indices, depth = dtype_indices_depth
    dtype, indices = dtype_and_indices
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=2,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="one_hot",
        indices=np.asarray(indices, dtype=dtype),
        depth=depth,
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
    num_positional_args=st.integers(0, 3),
)
def test_cumsum(
    dtype_x_axis,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="cumsum",
        x=np.asarray(x, dtype=dtype),
        axis=axis,
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
    num_positional_args=helpers.num_positional_args(fn_name="cumprod"),
)
def test_cumprod(
    dtype_x_axis,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    dtype, x, axis = dtype_x_axis
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="cumprod",
        x=np.asarray(x, dtype=dtype),
        axis=axis,
    )


# scatter_flat
@handle_cmd_line_args
@given(
    x=st.integers(min_value=1, max_value=10).flatmap(
        lambda n: st.tuples(
            helpers.dtype_and_values(
                available_dtypes=ivy_np.valid_float_dtypes,
                min_num_dims=1,
                max_num_dims=1,
                min_dim_size=n,
                max_dim_size=n,
            ),
            helpers.dtype_and_values(
                available_dtypes=ivy_np.valid_int_dtypes,
                min_value=0,
                max_value=max(n - 1, 0),
                min_num_dims=1,
                max_num_dims=1,
                min_dim_size=n,
                max_dim_size=n,
            ).filter(lambda l: len(set(l[1])) == len(l[1])),
            st.integers(min_value=n, max_value=n),
        )
    ),
    reduction=st.sampled_from(["sum", "min", "max", "replace"]),
    num_positional_args=st.integers(0, 3),
)
def test_scatter_flat(
    x,
    reduction,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    (val_dtype, vals), (ind_dtype, ind), size = x
    if fw == "torch" and (
        val_dtype in ["uint16", "uint32", "uint64"]
        or ind_dtype in ["uint16", "uint32", "uint64"]
    ):
        return
    helpers.test_function(
        input_dtypes=[ind_dtype, val_dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="scatter_flat",
        indices=np.asarray(ind, dtype=ind_dtype),
        updates=np.asarray(vals, dtype=val_dtype),
        size=size,
        reduction=reduction,
    )


# scatter_nd
@handle_cmd_line_args
@given(
    x=st.tuples(
        st.integers(min_value=2, max_value=5), st.integers(min_value=2, max_value=10)
    ).flatmap(
        lambda n: st.tuples(
            helpers.dtype_and_values(
                available_dtypes=ivy_np.valid_numeric_dtypes,
                min_num_dims=n[0],
                max_num_dims=n[0],
                min_dim_size=n[1],
                max_dim_size=n[1],
            ),
            helpers.dtype_and_values(
                available_dtypes=["int32", "int64"],
                min_value=0,
                max_value=max(n[1] - 1, 0),
                min_num_dims=1,
                max_num_dims=1,
                min_dim_size=n[1],
                max_dim_size=n[1],
                shape=st.shared(
                    helpers.get_shape(
                        min_num_dims=1,
                        max_num_dims=1,
                        min_dim_size=n[1],
                        max_dim_size=n[1],
                    ),
                    key="shape2",
                ),
            ).filter(lambda l: len(set(l[1])) == len(l[1])),
        )
    ),
    reduction=st.sampled_from(["sum", "min", "max", "replace"]),
    num_positional_args=helpers.num_positional_args(fn_name="scatter_nd"),
)
def test_scatter_nd(
    x,
    reduction,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    (val_dtype, vals), (ind_dtype, ind) = x
    shape = np.array(vals, dtype=val_dtype).shape
    helpers.test_function(
        input_dtypes=[ind_dtype, val_dtype],
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="scatter_nd",
        indices=np.asarray(ind, dtype=ind_dtype).reshape([len(vals), 1]),
        updates=np.asarray(vals, dtype=val_dtype),
        shape=shape,
        reduction=reduction,
    )


# gather
# @given(
#     params_n_indices_n_axis=helpers.array_and_indices_and_axis(
#         last_dim_same_size=False,
#         allow_inf=False,
#         min_num_dims=1,
#         max_num_dims=5,
#         min_dim_size=1,
#         max_dim_size=10
#     ),
#     as_variable=helpers.list_of_length(x=st.booleans(), length=2),
#     with_out=st.booleans(),
#     num_positional_args=helpers.num_positional_args(fn_name='gather'),
#     native_array=helpers.list_of_length(x=st.booleans(), length=2),
#     container=helpers.list_of_length(x=st.booleans(), length=2),
#     instance_method=st.booleans(),
# )
# def test_gather(params_n_indices_n_axis, as_variable, with_out,
#                 num_positional_args, native_array, container, instance_method , fw):
#     params, indices, axis = params_n_indices_n_axis
#     params_dtype, params = params
#     indices_dtype, indices = indices
#     helpers.test_function(
#         input_dtypes=[params_dtype, indices_dtype],
#         as_variable_flags=as_variable,
#         with_out=with_out,
#         num_positional_args=num_positional_args,
#         native_array_flags=native_array,
#         container_flags=container,
#         instance_method=instance_method,
#         fw=fw,
#         fn_name="gather",
#         params=np.asarray(params, dtype=params_dtype),
#         indices=np.asarray(indices, dtype=indices_dtype),
#         axis=axis
#     )


# gather_nd
# @given(
#     params_n_ndindices=helpers.array_and_ndindices(
#         allow_inf=False,
#         min_num_dims=1,
#         max_num_dims=5,
#         min_dim_size=1,
#         max_dim_size=10
#     ),
#     ndindices_dtype=st.sampled_from(["int32", "int64"]),
#     as_variable=helpers.list_of_length(st.booleans(), 2),
#     with_out=st.booleans(),
#     num_positional_args=helpers.num_positional_args(fn_name='gather_nd'),
#     native_array=helpers.list_of_length(st.booleans(), 2),
#     container=helpers.list_of_length(st.booleans(), 2),
#     instance_method=st.booleans(),
# )
# def test_gather_nd(params_n_ndindices, ndindices_dtype, as_variable,
#         with_out, num_positional_args, native_array, container, instance_method , fw):
#     params, ndindices = params_n_ndindices
#     params_dtype, params = params
#     helpers.test_function(
#         input_dtypes=[params_dtype, ndindices_dtype],
#         as_variable_flags=as_variable,
#         with_out=with_out,
#         num_positional_args=num_positional_args,
#         native_array_flags=native_array,
#         container_flags=container,
#         instance_method=instance_method,
#         fw=fw,
#         fn_name="gather_nd",
#         params=np.asarray(params, dtype=params_dtype),
#         indices=np.asarray(ndindices, dtype=ndindices_dtype)
#     )

# exists


@handle_cmd_line_args
@given(
    x=st.one_of(
        st.none(),
        helpers.dtype_and_values(
            available_dtypes=ivy_np.valid_numeric_dtypes,
            allow_inf=False,
            min_num_dims=0,
            min_dim_size=1,
        ),
        st.sampled_from([ivy.array]),
    )
)
def test_exists(x):
    if x is not None:
        if not hasattr(x, "__call__"):
            dtype, x = x
    ret = ivy.exists(x)
    assert isinstance(ret, bool)
    y_true = x is not None
    assert ret == y_true


# default
@handle_cmd_line_args
@given(
    x=st.one_of(
        st.none(),
        helpers.dtype_and_values(
            available_dtypes=ivy_np.valid_numeric_dtypes,
            allow_inf=False,
            min_num_dims=0,
            min_dim_size=2,
        ),
        st.sampled_from([ivy.array]),
    ),
    default_val=st.one_of(
        helpers.dtype_and_values(
            available_dtypes=ivy_np.valid_numeric_dtypes,
            allow_inf=False,
            min_num_dims=0,
            min_dim_size=2,
        ),
        st.sampled_from([ivy.array]),
    ),
)
def test_default(x, default_val):
    with_callable = False
    if x is not None:
        if hasattr(x, "__call__"):
            with_callable = True
        else:
            x_dtype, x = x
    else:
        if hasattr(default_val, "__call__"):
            with_callable = True
        else:
            dv_dtype, default_val = default_val

    truth_val = ivy.to_native(x if x is not None else default_val)
    if with_callable:
        assert ivy.default(x, default_val) == truth_val
    else:
        assert np.allclose(ivy.default(x, default_val), truth_val)


@handle_cmd_line_args
def test_cache_fn(device):
    def func():
        return ivy.random_uniform()

    # return a single cached_fn and then query this
    cached_fn = ivy.cache_fn(func)
    ret0 = cached_fn()
    ret0_again = cached_fn()
    ret1 = func()

    assert ivy.to_numpy(ret0).item() == ivy.to_numpy(ret0_again).item()
    assert ivy.to_numpy(ret0).item() != ivy.to_numpy(ret1).item()
    assert ret0 is ret0_again
    assert ret0 is not ret1

    # call ivy.cache_fn repeatedly, the new cached functions
    # each use the same global dict
    ret0 = ivy.cache_fn(func)()
    ret0_again = ivy.cache_fn(func)()
    ret1 = func()

    assert ivy.to_numpy(ret0).item() == ivy.to_numpy(ret0_again).item()
    assert ivy.to_numpy(ret0).item() != ivy.to_numpy(ret1).item()
    assert ret0 is ret0_again
    assert ret0 is not ret1


@handle_cmd_line_args
def test_cache_fn_with_args(device):
    def func(_):
        return ivy.random_uniform()

    # return a single cached_fn and then query this
    cached_fn = ivy.cache_fn(func)
    ret0 = cached_fn(0)
    ret0_again = cached_fn(0)
    ret1 = cached_fn(1)

    assert ivy.to_numpy(ret0).item() == ivy.to_numpy(ret0_again).item()
    assert ivy.to_numpy(ret0).item() != ivy.to_numpy(ret1).item()
    assert ret0 is ret0_again
    assert ret0 is not ret1

    # call ivy.cache_fn repeatedly, the new cached functions
    # each use the same global dict
    ret0 = ivy.cache_fn(func)(0)
    ret0_again = ivy.cache_fn(func)(0)
    ret1 = ivy.cache_fn(func)(1)

    assert ivy.to_numpy(ret0).item() == ivy.to_numpy(ret0_again).item()
    assert ivy.to_numpy(ret0).item() != ivy.to_numpy(ret1).item()
    assert ret0 is ret0_again
    assert ret0 is not ret1


@handle_cmd_line_args
def test_framework_setting_with_threading(device):
    if ivy.current_backend_str() == "jax":
        # Numpy is the conflicting framework being tested against
        pytest.skip()

    def thread_fn():
        x_ = jnp.array([0.0, 1.0, 2.0])
        ivy.set_backend("jax")
        for _ in range(2000):
            try:
                ivy.mean(x_)
            except TypeError:
                return False
        ivy.unset_backend()
        return True

    # get original framework string and array
    fws = ivy.current_backend_str()
    x = ivy.array([0.0, 1.0, 2.0])

    # start jax loop thread
    thread = threading.Thread(target=thread_fn)
    thread.start()
    time.sleep(0.01)
    # start local original framework loop
    ivy.set_backend(fws)
    for _ in range(2000):
        ivy.mean(x)
    ivy.unset_backend()
    assert not thread.join()


@handle_cmd_line_args
def test_framework_setting_with_multiprocessing(device):
    if ivy.current_backend_str() == "numpy":
        # Numpy is the conflicting framework being tested against
        pytest.skip()

    def worker_fn(out_queue):
        ivy.set_backend("numpy")
        x_ = np.array([0.0, 1.0, 2.0])
        for _ in range(1000):
            try:
                ivy.mean(x_)
            except TypeError:
                out_queue.put(False)
                return
        ivy.unset_backend()
        out_queue.put(True)

    # get original framework string and array
    fws = ivy.current_backend_str()
    x = ivy.array([0.0, 1.0, 2.0])

    # start numpy loop thread
    output_queue = multiprocessing.Queue()
    worker = multiprocessing.Process(target=worker_fn, args=(output_queue,))
    worker.start()

    # start local original framework loop
    ivy.set_backend(fws)
    for _ in range(1000):
        ivy.mean(x)
    ivy.unset_backend()

    worker.join()
    assert output_queue.get_nowait()


@handle_cmd_line_args
def test_explicit_ivy_framework_handles(device):
    if ivy.current_backend_str() == "numpy":
        # Numpy is the conflicting framework being tested against
        pytest.skip()

    # store original framework string and unset
    fw_str = ivy.current_backend_str()
    ivy.unset_backend()

    # set with explicit handle caught
    ivy_exp = ivy.get_backend(fw_str)
    assert ivy_exp.current_backend_str() == fw_str

    # assert backend implemented function is accessible
    assert "array" in ivy_exp.__dict__
    assert callable(ivy_exp.array)

    # assert joint implemented function is also accessible
    assert "cache_fn" in ivy_exp.__dict__
    assert callable(ivy_exp.cache_fn)

    # set global ivy to numpy
    ivy.set_backend("numpy")

    # assert the explicit handle is still unchanged
    assert ivy.current_backend_str() == "numpy"
    assert ivy_exp.current_backend_str() == fw_str

    # unset global ivy from numpy
    ivy.unset_backend()


# ToDo: re-add this test once ivy.get_backend is working correctly, with the returned
#  ivy handle having no dependence on the globally set ivy
# @handle_cmd_line_args
#
# def test_class_ivy_handles(device, call):
#
#     if call is helpers.np_call:
#         # Numpy is the conflicting framework being tested against
#         pytest.skip()
#
#     class ArrayGen:
#         def __init__(self, ivyh):
#             self._ivy = ivyh
#
#         def get_array(self):
#             return self._ivy.array([0.0, 1.0, 2.0], dtype="float32", device=device)
#
#     # create instance
#     ag = ArrayGen(ivy.get_backend())
#
#     # create array from array generator
#     x = ag.get_array()
#
#     # verify this is not a numpy array
#     assert not isinstance(x, np.ndarray)
#
#     # change global framework to numpy
#     ivy.set_backend("numpy")
#
#     # create another array from array generator
#     x = ag.get_array()
#
#     # verify this is not still a numpy array
#     assert not isinstance(x, np.ndarray)


# einops_rearrange
@handle_cmd_line_args
@given(
    x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        allow_inf=False,
        min_num_dims=3,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=2,
    ).filter(
        lambda x: (ivy.array([x[1]], dtype="float32").shape[2] % 2 == 0)
        and (ivy.array([x[1]], dtype="float32").shape[3] % 2 == 0)
    ),
    pattern_and_axes_lengths=st.sampled_from(
        [
            ("b h w c -> b h w c", {}),
            ("b h w c -> (b h) w c", {}),
            ("b h w c -> b c h w", {}),
            ("b h w c -> h (b w) c", {}),
            ("b h w c -> b (c h w)", {}),
            ("b (h1 h) (w1 w) c -> (b h1 w1) h w c", {"h1": 2, "w1": 2}),
            ("b (h h1) (w w1) c -> b h w (c h1 w1)", {"h1": 2, "w1": 2}),
        ]
    ),
    num_positional_args=helpers.num_positional_args(fn_name="einops_rearrange"),
)
def test_einops_rearrange(
    x,
    pattern_and_axes_lengths,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    pattern, axes_lengths = pattern_and_axes_lengths
    dtype, x = x
    x = [x]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="einops_rearrange",
        x=np.asarray(x, dtype=dtype),
        pattern=pattern,
        **axes_lengths,
    )


@handle_cmd_line_args
@given(
    x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        allow_inf=False,
        min_num_dims=3,
        max_num_dims=3,
        min_dim_size=2,
        max_dim_size=2,
    ).filter(
        lambda x: ivy.array([x[1]], dtype="float32").shape[2] % 2 == 0
        and ivy.array([x[1]], dtype="float32").shape[3] % 2 == 0
    ),
    pattern_and_axes_lengths=st.sampled_from(
        [
            # ('t b  -> b', {}),
            ("b c (h1 h2) (w1 w2) -> b c h1 w1", {"h2": 2, "w2": 2}),
        ]
    ),
    reduction=st.sampled_from(["min", "max", "sum", "mean", "prod"]),
    num_positional_args=helpers.num_positional_args(fn_name="einops_reduce"),
)
def test_einops_reduce(
    x,
    pattern_and_axes_lengths,
    reduction,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
):
    pattern, axes_lengths = pattern_and_axes_lengths
    dtype, x = x
    x = [x]
    if (reduction in ["mean", "prod"]) and (dtype not in ivy_np.valid_float_dtypes):
        dtype = "float32"
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="einops_reduce",
        x=np.asarray(x, dtype=dtype),
        pattern=pattern,
        reduction=reduction,
        **axes_lengths,
    )


# einops_repeat
@handle_cmd_line_args
@given(
    x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
    ),
    pattern_and_axes_lengths=st.sampled_from(
        [
            ("h w -> h w repeat", {"repeat": 2}),
            ("h w -> (repeat h) w", {"repeat": 2}),
            ("h w -> h (repeat w)", {"repeat": 2}),
            ("h w -> (h h2) (w w2)", {"h2": 2, "w2": 2}),
            ("h w  -> w h", {}),
        ]
    ),
    num_positional_args=helpers.num_positional_args(fn_name="einops_repeat"),
)
def test_einops_repeat(
    x,
    pattern_and_axes_lengths,
    with_out,
    as_variable,
    num_positional_args,
    native_array,
    container,
    instance_method,
    fw,
    device,
):
    pattern, axes_lengths = pattern_and_axes_lengths
    dtype, x = x
    x = [x]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="einops_repeat",
        x=np.asarray(x, dtype=dtype),
        pattern=pattern,
        **axes_lengths,
    )


# container types
@handle_cmd_line_args
def test_container_types(device):
    cont_types = ivy.container_types()
    assert isinstance(cont_types, list)
    for cont_type in cont_types:
        assert hasattr(cont_type, "keys")
        assert hasattr(cont_type, "values")
        assert hasattr(cont_type, "items")


@handle_cmd_line_args
def test_inplace_arrays_supported(device):
    cur_fw = ivy.current_backend_str()
    if cur_fw in ["numpy", "mxnet", "torch"]:
        assert ivy.inplace_arrays_supported()
    elif cur_fw in ["jax", "tensorflow"]:
        assert not ivy.inplace_arrays_supported()
    else:
        raise Exception("Unrecognized framework")


@handle_cmd_line_args
def test_inplace_variables_supported(device):
    cur_fw = ivy.current_backend_str()
    if cur_fw in ["numpy", "mxnet", "torch", "tensorflow"]:
        assert ivy.inplace_variables_supported()
    elif cur_fw in ["jax"]:
        assert not ivy.inplace_variables_supported()
    else:
        raise Exception("Unrecognized framework")


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        num_arrays=2,
        shared_dtype=True,
    ),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
)
def test_inplace_update(x_val_and_dtypes, tensor_fn, device):
    # ToDo: Ask Daniel about tensor_fn, we use it here since
    #  we don't use helpers.test_function
    x, val = x_val_and_dtypes[1]
    x = tensor_fn(x, dtype="float32", device=device)
    val = tensor_fn(val, dtype="float32", device=device)
    if (tensor_fn is not helpers.var_fn and ivy.inplace_arrays_supported()) or (
        tensor_fn is helpers.var_fn and ivy.inplace_variables_supported()
    ):
        x_inplace = ivy.inplace_update(x, val)
        assert id(x_inplace) == id(x)
        assert np.allclose(ivy.to_numpy(x), ivy.to_numpy(val))
        return


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        num_arrays=2,
        shared_dtype=True,
    ),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
)
def test_inplace_decrement(x_val_and_dtypes, tensor_fn, device):
    x, val = x_val_and_dtypes[1]
    x = tensor_fn(x, dtype="float32", device=device)
    val = tensor_fn(val, dtype="float32", device=device)
    new_val = x - val
    if (tensor_fn is not helpers.var_fn and ivy.inplace_arrays_supported()) or (
        tensor_fn is helpers.var_fn and ivy.inplace_variables_supported()
    ):
        x_inplace = ivy.inplace_decrement(x, val)
        assert id(x_inplace) == id(x)
        assert np.allclose(ivy.to_numpy(new_val), ivy.to_numpy(x))
        return


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        allow_inf=False,
        min_num_dims=1,
        max_num_dims=1,
        min_dim_size=2,
        num_arrays=2,
        shared_dtype=True,
    ),
    tensor_fn=st.sampled_from([ivy.array, helpers.var_fn]),
)
def test_inplace_increment(x_val_and_dtypes, tensor_fn, device):
    x, val = x_val_and_dtypes[1]
    x = tensor_fn(x, dtype="float32", device=device)
    val = tensor_fn(val, dtype="float32", device=device)
    new_val = x + val
    if (tensor_fn is not helpers.var_fn and ivy.inplace_arrays_supported()) or (
        tensor_fn is helpers.var_fn and ivy.inplace_variables_supported()
    ):
        x_inplace = ivy.inplace_increment(x, val)
        assert id(x_inplace) == id(x)
        assert np.allclose(ivy.to_numpy(new_val), ivy.to_numpy(x_inplace))
        return


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes),
    exclusive=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="is_ivy_array"),
)
def test_is_ivy_array(
    x_val_and_dtypes,
    exclusive,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    dtype = x_val_and_dtypes[0]
    x = x_val_and_dtypes[1]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=False,
        fw=fw,
        fn_name="is_ivy_array",
        x=np.asarray(x, dtype=dtype),
        exclusive=exclusive,
    )


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes),
    exclusive=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="is_array"),
)
def test_is_array(
    x_val_and_dtypes,
    exclusive,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    dtype = x_val_and_dtypes[0]
    x = x_val_and_dtypes[1]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=False,
        fw=fw,
        fn_name="is_array",
        x=np.asarray(x, dtype=dtype),
        exclusive=exclusive,
    )


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="is_ivy_container"),
)
def test_is_ivy_container(
    x_val_and_dtypes, as_variable, num_positional_args, native_array, container, fw
):
    dtype = x_val_and_dtypes[0]
    x = x_val_and_dtypes[1]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=False,
        fw=fw,
        fn_name="is_ivy_container",
        x=np.asarray(x, dtype=dtype),
    )


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_dtypes, num_arrays=2, min_num_dims=1
    ),
    equality_matrix=st.booleans(),
    num_positional_args=st.integers(min_value=2, max_value=3),
)
def test_all_equal(
    x_val_and_dtypes,
    equality_matrix,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    dtype = x_val_and_dtypes[0]
    x = x_val_and_dtypes[1]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=False,
        fw=fw,
        fn_name="all_equal",
        x0=np.asarray(x[0], dtype=dtype[0]),
        x1=np.asarray(x[1], dtype=dtype[1]),
        equality_matrix=equality_matrix,
    )


@handle_cmd_line_args
@given(
    x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=1,
        max_dim_size=5,
        min_value=-10,
        max_value=10,
    ),
    max_norm=st.floats(min_value=0.137, max_value=1e05),
    p=st.sampled_from([1, 2, float("inf"), "fro", "nuc"]),
)
def test_clip_matrix_norm(
    x,
    max_norm,
    p,
    as_variable,
    with_out,
    native_array,
    container,
    instance_method,
    device,
    fw,
):
    dtype, x = x[0], x[1]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=2,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=instance_method,
        fw=fw,
        fn_name="clip_matrix_norm",
        rtol_=1e-2,
        atol_=1e-2,
        x=np.asarray(x, dtype=dtype),
        max_norm=max_norm,
        p=p,
    )


@handle_cmd_line_args
@given(
    x_n_include_inf_n_value=st.sampled_from(
        [
            [ivy.array([1]), True, False],
            [ivy.array(ivy.nan), False, True],
            [ivy.native_array(ivy.inf), True, True],
            [ivy.array(ivy.inf), False, False],
        ]
    )
)
def test_value_is_nan(x_n_include_inf_n_value):
    x, include_inf, value = x_n_include_inf_n_value
    ret = ivy.value_is_nan(x, include_inf)
    assert ret == value


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes),
    include_infs=st.booleans(),
    num_positional_args=helpers.num_positional_args(fn_name="has_nans"),
)
def test_has_nans(
    x_val_and_dtypes,
    include_infs,
    as_variable,
    num_positional_args,
    native_array,
    container,
    fw,
):
    dtype = x_val_and_dtypes[0]
    x = x_val_and_dtypes[1]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=True,
        fw=fw,
        fn_name="has_nans",
        x=np.asarray(x, dtype=dtype),
        include_infs=include_infs,
    )


@handle_cmd_line_args
@given(
    x=st.booleans(),
)
def test_try_else_none(x):
    if x:
        fn = ivy.try_else_none(lambda: True)
        assert fn() is True
    else:
        fn = ivy.try_else_none(lambda x: x)
        assert fn is None


@handle_cmd_line_args
@given(
    x_n_value=st.sampled_from(
        [
            [ivy.value_is_nan, ["x", "include_infs"]],
            [ivy.clip_matrix_norm, ["x", "max_norm", "p", "out"]],
        ]
    )
)
def test_arg_names(x_n_value):
    x, value = x_n_value
    ret = ivy.arg_names(x)
    assert ret == value


def _composition_1():
    return ivy.relu().argmax()


def _composition_2():
    return ivy.ceil() or ivy.linspace()


# function_supported_devices_and_dtypes
@pytest.mark.parametrize(
    "func, expected",
    [
        (
            _composition_1,
            {
                "cpu": (
                    "bool",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "bfloat16",
                    "float16",
                    "float32",
                    "float64",
                )
            },
        ),
        (
            _composition_2,
            {
                "cpu": (
                    "bool",
                    "uint8",
                    "uint16",
                    "uint32",
                    "uint64",
                    "int8",
                    "int16",
                    "int32",
                    "int64",
                    "bfloat16",
                    "float16",
                    "float32",
                    "float64",
                )
            },
        ),
    ],
)
def test_function_supported_device_and_dtype(func, expected):
    res = ivy.function_supported_devices_and_dtypes(func)
    exp = {}
    for dev in expected:
        exp[dev] = tuple((set(ivy.valid_dtypes).intersection(expected[dev])))
        if ivy.current_backend_str() == "torch":
            exp[dev] = tuple((set(exp[dev]).difference({"float16"})))

    all_key = set(res.keys()).union(set(exp.keys()))
    for key in all_key:
        assert key in res
        assert key in exp
        assert sorted(res[key]) == sorted(exp[key])


# function_unsupported_devices_and_dtypes
@pytest.mark.parametrize(
    "func, expected",
    [
        (_composition_1, {"gpu": ivy.all_dtypes, "tpu": ivy.all_dtypes}),
        # (_composition_2, {'gpu': ivy.all_dtypes, 'tpu': ivy.all_dtypes})
    ],
)
def test_function_unsupported_devices(func, expected):
    res = ivy.function_unsupported_devices_and_dtypes(func)

    exp = expected.copy()

    if ivy.invalid_dtypes:
        exp["cpu"] = ivy.invalid_dtypes
    if ivy.current_backend_str() == "torch":
        exp["cpu"] = tuple((set(exp["cpu"]).union({"float16"})))

    all_key = set(res.keys()).union(set(exp.keys()))
    for key in all_key:
        assert key in res
        assert key in exp
        assert sorted(res[key]) == sorted(exp[key])


# Still to Add #
# ---------------#


@handle_cmd_line_args
def test_current_backend_str(fw):
    assert ivy.current_backend_str() == fw


@handle_cmd_line_args
def test_get_min_denominator():
    assert ivy.get_min_denominator() == 1e-12


@handle_cmd_line_args
@given(x=st.floats(allow_nan=False, allow_infinity=False))
def test_set_min_denominator(x):
    ivy.set_min_denominator(x)
    assert ivy.get_min_denominator() == x


@handle_cmd_line_args
def test_get_min_base():
    assert ivy.get_min_base() == 1e-5


@handle_cmd_line_args
@given(x=st.floats(allow_nan=False, allow_infinity=False))
def test_set_min_base(x):
    ivy.set_min_base(x)
    assert ivy.get_min_base() == x


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes, num_arrays=3, shared_dtype=True
    ),
    num_positional_args=helpers.num_positional_args(fn_name="stable_divide"),
)
def test_stable_divide(
    dtype_and_x, as_variable, num_positional_args, native_array, container, fw
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=False,
        fw=fw,
        fn_name="stable_divide",
        numerator=np.asarray(x[0], dtype=input_dtype[0]),
        denominator=np.asarray(x[1], dtype=input_dtype[1]),
        min_denominator=np.asarray(x[2], dtype=input_dtype[2]),
    )


@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=ivy_np.valid_numeric_dtypes,
        num_arrays=3,
        min_value=0,
        shared_dtype=True,
    ),
    num_positional_args=helpers.num_positional_args(fn_name="stable_pow"),
)
def test_stable_pow(
    dtype_and_x, as_variable, num_positional_args, native_array, container, fw
):
    input_dtype, x = dtype_and_x
    for i in range(len(input_dtype)):
        if input_dtype[i] in ["uint8", "uint16", "uint32", "uint64"]:
            return
    # assume(not (input_dtype[i] in ["uint8"] for i in range(len(input_dtype))))
    helpers.test_function(
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=False,
        fw=fw,
        fn_name="stable_pow",
        base=np.asarray(x[0], dtype=input_dtype[0]),
        exponent=np.asarray(x[1], dtype=input_dtype[1]),
        min_base=np.asarray(x[2], dtype=input_dtype[2]),
    )


@handle_cmd_line_args
def test_get_all_arrays_in_memory():
    return


@handle_cmd_line_args
def test_num_arrays_in_memory():
    return


@handle_cmd_line_args
def test_print_all_arrays_in_memory():
    return


@handle_cmd_line_args
@given(
    x=st.floats(allow_nan=False, allow_infinity=False),
)
def test_set_queue_timeout(x):
    ivy.set_queue_timeout(x)
    ret = ivy.get_queue_timeout()
    assert ret == x


@handle_cmd_line_args
@given(
    x=st.floats(allow_nan=False, allow_infinity=False),
)
def test_get_queue_timeout(x):
    ivy.set_queue_timeout(x)
    ret = ivy.get_queue_timeout()
    assert ret == x


@handle_cmd_line_args
def test_get_tmp_dir():
    ret = ivy.get_tmp_dir()
    assert ret == "/tmp"


@handle_cmd_line_args
def test_set_tmp_dir():
    ivy.set_tmp_dir("/new_dir")
    ret = ivy.get_tmp_dir()
    assert ret == "/new_dir"


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="supports_inplace"),
)
def test_supports_inplace(
    x_val_and_dtypes, as_variable, num_positional_args, native_array, container, fw
):
    dtype = x_val_and_dtypes[0]
    x = x_val_and_dtypes[1]
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=True,
        fw=fw,
        fn_name="supports_inplace",
        x=np.asarray(x, dtype=dtype),
    )


@handle_cmd_line_args
@given(
    x_val_and_dtypes=helpers.dtype_and_values(available_dtypes=ivy_np.valid_dtypes),
    num_positional_args=helpers.num_positional_args(fn_name="assert_supports_inplace"),
)
def test_assert_supports_inplace(
    x_val_and_dtypes, as_variable, num_positional_args, native_array, container, fw
):
    dtype = x_val_and_dtypes[0]
    x = x_val_and_dtypes[1]
    if fw == "tensorflow" or fw == "jax":
        return
    helpers.test_function(
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container,
        instance_method=True,
        fw=fw,
        fn_name="assert_supports_inplace",
        ground_truth_backend="numpy",
        x=np.asarray(x, dtype=dtype),
    )


@handle_cmd_line_args
def test_arg_info():
    return


def _fn1(x, y):
    return ivy.matmul(x, y)


def _fn2(x, y):
    return ivy.vecdot(x, y)


def _fn3(x, y):
    ivy.add(x, y)


@given(
    func=st.sampled_from([_fn1, _fn2, _fn3]),
    arrays_and_axes=helpers.arrays_and_axes(
        allow_none=False,
        min_num_dims=2,
        max_num_dims=5,
        min_dim_size=2,
        max_dim_size=10,
        num=2,
    ),
    in_axes_as_cont=st.booleans(),
)
def test_vmap(func, arrays_and_axes, in_axes_as_cont):

    generated_arrays, in_axes = arrays_and_axes
    arrays = [ivy.native_array(array) for array in generated_arrays]

    if in_axes_as_cont:
        vmapped_func = ivy.vmap(func, in_axes=in_axes, out_axes=0)
    else:
        vmapped_func = ivy.vmap(func, in_axes=0, out_axes=0)

    assert callable(vmapped_func)

    try:
        fw_res = vmapped_func(*arrays)
    except Exception:
        fw_res = None

    ivy.set_backend("jax")
    arrays = [ivy.native_array(array) for array in generated_arrays]
    if in_axes_as_cont:
        jax_vmapped_func = ivy.vmap(func, in_axes=in_axes, out_axes=0)
    else:
        jax_vmapped_func = ivy.vmap(func, in_axes=0, out_axes=0)

    assert callable(jax_vmapped_func)

    try:
        jax_res = jax_vmapped_func(*arrays)
    except Exception:
        jax_res = None

    ivy.unset_backend()

    if fw_res is not None and jax_res is not None:
        assert np.allclose(
            fw_res, jax_res
        ), f"Results from {ivy.current_backend_str()} and jax are not equal"

    elif fw_res is None and jax_res is None:
        pass
    else:
        assert False, "One of the results is None while other isn't"

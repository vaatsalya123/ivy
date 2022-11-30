"""Collection of tests for unified dtype functions."""

# global
import numpy as np
import importlib
from hypothesis import strategies as st
import typing
from types import SimpleNamespace

# local
import ivy

try:
    import ivy.functional.backends.tensorflow as ivy_tf
except ImportError:
    ivy_tf = SimpleNamespace()
    ivy_tf.valid_dtypes = ()
    ivy_tf.invalid_dtypes = ()

try:
    import ivy.functional.backends.jax as ivy_jax
except ImportError:
    ivy_jax = SimpleNamespace()
    ivy_jax.valid_dtypes = ()
    ivy_jax.invalid_dtypes = ()

try:
    import ivy.functional.backends.torch as ivy_torch
except ImportError:
    ivy_torch = SimpleNamespace()
    ivy_torch.valid_dtypes = ()
    ivy_torch.invalid_dtypes = ()

import ivy.functional.backends.numpy as ivy_np
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


# dtype objects
@handle_test(fn_tree="functional.ivy.exists")  # dummy fn_tree
def test_dtype_instances():
    assert ivy.exists(ivy.int8)
    assert ivy.exists(ivy.int16)
    assert ivy.exists(ivy.int32)
    assert ivy.exists(ivy.int64)
    assert ivy.exists(ivy.uint8)
    if ivy.current_backend_str() != "torch":
        assert ivy.exists(ivy.uint16)
        assert ivy.exists(ivy.uint32)
        assert ivy.exists(ivy.uint64)
    assert ivy.exists(ivy.float32)
    assert ivy.exists(ivy.float64)
    assert ivy.exists(ivy.bool)


# for data generation in multiple tests
dtype_shared = helpers.get_dtypes("valid", full=False, key="dtype")


@st.composite
def dtypes_shared(draw, num_dtypes):
    if isinstance(num_dtypes, str):
        num_dtypes = draw(st.shared(helpers.ints(), key=num_dtypes))
    return draw(
        st.shared(
            st.lists(
                st.sampled_from(draw(helpers.get_dtypes("valid"))),
                min_size=num_dtypes,
                max_size=num_dtypes,
            ),
            key="dtypes",
        )
    )


# Array API Standard Function Tests #
# --------------------------------- #


@st.composite
def astype_helper(draw):
    dtype, x = draw(
        helpers.dtype_and_values(
            available_dtypes=helpers.get_dtypes("valid"),
            num_arrays=1,
            small_abs_safety_factor=4,
            large_abs_safety_factor=4,
            safety_factor_scale="log",
        )
    )

    cast_dtype = draw(
        helpers.get_castable_dtype(draw(helpers.get_dtypes("valid")), dtype[0], x)
    )
    return dtype, x, cast_dtype


# astype
@handle_test(
    fn_tree="functional.ivy.astype",
    dtype_and_x_and_cast_dtype=astype_helper(),
)
def test_astype(
    *,
    dtype_and_x_and_cast_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x, cast_dtype = dtype_and_x_and_cast_dtype
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        rtol_=1e-3,
        atol_=1e-3,
        x=x[0],
        dtype=cast_dtype[0],
    )


# broadcast arrays
@st.composite
def broadcastable_arrays(draw, dtypes):
    num_arrays = st.shared(helpers.ints(min_value=2, max_value=5), key="num_arrays")
    shapes = draw(num_arrays.flatmap(helpers.mutually_broadcastable_shapes))
    dtypes = draw(dtypes)
    arrays = []
    for c, (shape, dtype) in enumerate(zip(shapes, dtypes), 1):
        x = draw(helpers.array_values(dtype=dtype, shape=shape), label=f"x{c}").tolist()
        arrays.append(x)
    return arrays


@handle_test(
    fn_tree="functional.ivy.broadcast_arrays",
    arrays=broadcastable_arrays(dtypes_shared("num_arrays")),
    input_dtypes=dtypes_shared("num_arrays"),
)
def test_broadcast_arrays(
    *,
    arrays,
    input_dtypes,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    if backend_fw.current_backend_str() == "torch":
        for input_dtype in input_dtypes:
            if input_dtype == "bfloat16" or (
                "uint" in input_dtype and "uint8" not in input_dtype
            ):
                # Torch has no inference strategy for bfloat16
                # Torch has no support for uint above uint8
                return

    kw = {}
    for i, (array, dtype) in enumerate(zip(arrays, input_dtypes)):
        kw["x{}".format(i)] = np.asarray(array, dtype=dtype)
    num_positional_args = len(kw)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtypes,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        **kw,
    )


@handle_test(
    fn_tree="functional.ivy.broadcast_to",
    array_and_shape=helpers.array_and_broadcastable_shape(dtype_shared),
    input_dtype=dtype_shared,
)
def test_broadcast_to(
    *,
    array_and_shape,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    if backend_fw.current_backend_str() == "torch":
        if input_dtype == "bfloat16" or (
            "uint" in input_dtype and "uint8" not in input_dtype
        ):
            # Torch has no inference strategy for bfloat16
            # Torch has no support for uint above uint8
            return

    array, to_shape = array_and_shape
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=array,
        shape=to_shape,
    )


# can_cast
@handle_test(
    fn_tree="functional.ivy.can_cast",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"), num_arrays=1
    ),
    to_dtype=helpers.get_dtypes("valid", full=False),
)
def test_can_cast(
    *,
    dtype_and_x,
    to_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    input_dtype, x = dtype_and_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        from_=x[0],
        to=to_dtype[0],
    )


@st.composite
def _array_or_type(draw, float_or_int):
    valid_dtypes = {
        "float": draw(helpers.get_dtypes("float")),
        "int": draw(helpers.get_dtypes("integer")),
    }[float_or_int]
    return draw(
        st.sampled_from(
            (
                draw(
                    helpers.dtype_and_values(
                        available_dtypes=valid_dtypes,
                    )
                ),
                draw(st.sampled_from(valid_dtypes)),
            )
        )
    )


# finfo
@handle_test(
    fn_tree="functional.ivy.finfo",
    type=_array_or_type("float"),
)
def test_finfo(
    *,
    type,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    if isinstance(type, str):
        input_dtype = [type]
    else:
        input_dtype, x = type
        type = x[0]
    ret = helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        type=type,
        test_values=False,
    )
    if not ivy.exists(ret):
        return
    mach_lims, mach_lims_np = ret
    assert np.allclose(mach_lims.min, mach_lims_np.min, rtol=1e-2, atol=1e-2)
    assert np.allclose(mach_lims.max, mach_lims_np.max, rtol=1e-2, atol=1e-2)
    assert np.allclose(mach_lims.eps, mach_lims_np.eps, rtol=1e-2, atol=1e-2)
    assert mach_lims.bits == mach_lims_np.bits


# iinfo
@handle_test(
    fn_tree="functional.ivy.iinfo",
    type=_array_or_type("int"),
)
def test_iinfo(
    *,
    type,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    if isinstance(type, str):
        input_dtype = [type]
    else:
        input_dtype, x = type
        type = x[0]
    ret = helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        type=type,
        test_values=False,
    )

    if not ivy.exists(ret):
        return
    mach_lims, mach_lims_np = ret
    assert mach_lims.min == mach_lims_np.min
    assert mach_lims.max == mach_lims_np.max
    assert mach_lims.bits == mach_lims_np.bits


# result_type
@handle_test(
    fn_tree="functional.ivy.result_type",
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=st.shared(helpers.ints(min_value=2, max_value=5), key="num_arrays"),
        shared_dtype=False,
    ),
)
def test_result_type(
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
    ground_truth_backend,
):
    dtype, x = helpers.as_lists(*dtype_and_x)
    kw = {}
    for i, (dtype_, x_) in enumerate(zip(dtype, x)):
        kw["x{}".format(i)] = x_
    num_positional_args = len(kw)
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        on_device=on_device,
        fn_name=fn_name,
        **kw,
    )


# Extra Ivy Function Tests #
# ------------------------ #


# as_ivy_dtype
@handle_test(
    fn_tree="functional.ivy.as_ivy_dtype",
    input_dtype=st.sampled_from(ivy.valid_dtypes),
)
def test_as_ivy_dtype(
    *,
    input_dtype,
    ground_truth_backend,
):
    res = ivy.as_ivy_dtype(input_dtype)
    if isinstance(input_dtype, str):
        assert isinstance(res, str)
        return

    assert isinstance(input_dtype, ivy.Dtype) or isinstance(
        input_dtype, str
    ), f"input_dtype={input_dtype!r}, but should be str or ivy.Dtype"
    assert isinstance(res, str), f"result={res!r}, but should be str"


_valid_dtype_in_all_frameworks = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "float16",
    "float32",
    "float64",
    "bool",
]


# as_native_dtype
@handle_test(
    fn_tree="functional.ivy.as_native_dtype",
    input_dtype=st.sampled_from(_valid_dtype_in_all_frameworks),
)
def test_as_native_dtype(
    *,
    input_dtype,
    with_out,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    res = ivy.as_native_dtype(input_dtype)
    if isinstance(input_dtype, ivy.NativeDtype):
        assert isinstance(res, ivy.NativeDtype)
        return

    assert isinstance(input_dtype, ivy.Dtype) or isinstance(
        input_dtype, str
    ), f"input_dtype={input_dtype!r}, but should be str or ivy.Dtype"
    assert isinstance(
        res, ivy.NativeDtype
    ), f"result={res!r}, but should be ivy.NativeDtype"


# closest_valid_dtypes
@handle_test(
    fn_tree="functional.ivy.closest_valid_dtype",
    input_dtype=st.sampled_from(_valid_dtype_in_all_frameworks),
)
def test_closest_valid_dtype(*, input_dtype, with_out, backend_fw, fn_name, on_device):
    res = ivy.closest_valid_dtype(input_dtype)
    assert isinstance(input_dtype, ivy.Dtype) or isinstance(input_dtype, str)
    assert isinstance(res, ivy.Dtype) or isinstance(
        res, str
    ), f"result={res!r}, but should be str or ivy.Dtype"


# default_dtype
@handle_test(
    fn_tree="functional.ivy.default_dtype",
    input_dtype=helpers.get_dtypes("valid", full=False),
)
def test_default_dtype(
    *, input_dtype, native_array, with_out, backend_fw, fn_name, on_device
):
    input_dtype = input_dtype[0]
    res = ivy.default_dtype(dtype=input_dtype, as_native=native_array)
    assert (
        isinstance(input_dtype, ivy.Dtype)
        or isinstance(input_dtype, str)
        or isinstance(input_dtype, ivy.NativeDtype)
    )
    assert isinstance(res, ivy.Dtype) or isinstance(
        input_dtype, str
    ), f"input_dtype={input_dtype!r}, but should be str or ivy.Dtype"


# dtype
# TODO: fix instance method
@handle_test(
    fn_tree="functional.ivy.dtype",
    array=helpers.array_values(
        dtype=dtype_shared,
        shape=helpers.lists(
            arg=helpers.ints(min_value=1, max_value=5),
            min_size="num_dims",
            max_size="num_dims",
            size_bounds=[1, 5],
        ),
    ),
    input_dtype=dtype_shared,
    as_native=st.booleans(),
)
def test_dtype(
    *,
    array,
    input_dtype,
    as_native,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    if backend_fw.current_backend_str() == "torch":
        if input_dtype == "bfloat16" or (
            "uint" in input_dtype and "uint8" not in input_dtype
        ):
            # Torch has no inference strategy for bfloat16
            # Torch has no support for uint above uint8
            return

    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=False,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x=array,
        as_native=as_native,
        test_values=False,
    )


# dtype_bits
# TODO: fix instance method
@handle_test(
    fn_tree="functional.ivy.dtype_bits",
    input_dtype=helpers.get_dtypes("valid", full=False),
)
def test_dtype_bits(
    *,
    input_dtype,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    ret = helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=False,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        dtype_in=input_dtype[0],
        test_values=False,
    )
    if not ivy.exists(ret):
        return
    num_bits, num_bits_np = ret
    assert num_bits == num_bits_np


# is_bool_dtype
@handle_test(
    fn_tree="functional.ivy.is_bool_dtype",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=False)
    ),
)
def test_is_bool_dtype(
    *,
    dtype_x,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        dtype_in=x[0],
    )


# is_float_dtype
@handle_test(
    fn_tree="functional.ivy.is_float_dtype",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=False)
    ),
)
def test_is_float_dtype(
    *,
    dtype_x,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        dtype_in=x[0],
    )


# is_int_dtype
@handle_test(
    fn_tree="functional.ivy.is_int_dtype",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=False)
    ),
)
def test_is_int_dtype(
    *,
    dtype_x,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        dtype_in=x[0],
    )


# is_uint_dtype
@handle_test(
    fn_tree="functional.ivy.is_uint_dtype",
    dtype_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid", full=False)
    ),
)
def test_is_uint_dtype(
    *,
    dtype_x,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, x = dtype_x
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        dtype_in=x[0],
    )


# promote_types
# TODO: fix instance method
@handle_test(
    fn_tree="functional.ivy.promote_types",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=False,
    ),
)
def test_promote_types(
    *,
    dtype_and_values,
    as_variable,
    num_positional_args,
    native_array,
    container_flags,
    with_out,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    types, arrays = dtype_and_values
    type1, type2 = types
    input_dtype = [type1, type2]
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=input_dtype,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=container_flags,
        instance_method=False,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        type1=type1,
        type2=type2,
        test_values=False,
    )


# type_promote_arrays
# TODO: fix container method
@handle_test(
    fn_tree="functional.ivy.type_promote_arrays",
    dtype_and_values=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=False,
    ),
)
def test_type_promote_arrays(
    *,
    dtype_and_values,
    as_variable,
    num_positional_args,
    native_array,
    with_out,
    instance_method,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    types, arrays = dtype_and_values
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        input_dtypes=types,
        as_variable_flags=as_variable,
        with_out=with_out,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        container_flags=[False],
        instance_method=instance_method,
        fw=backend_fw,
        fn_name=fn_name,
        on_device=on_device,
        x1=arrays[0],
        x2=arrays[1],
        test_values=True,
    )


# default_float_dtype
@handle_test(
    fn_tree="functional.ivy.default_float_dtype",
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("float")),
)
def test_default_float_dtype(
    *,
    dtype_x,
    native_array,
    with_out,
    backend_fw,
    fn_name,
    on_device,
):
    float_dtype, x = dtype_x
    res = ivy.default_float_dtype(
        input=input, float_dtype=float_dtype[0], as_native=native_array
    )
    assert (
        isinstance(res, ivy.Dtype)
        or isinstance(res, typing.get_args(ivy.NativeDtype))
        or isinstance(res, ivy.NativeDtype)
        or isinstance(res, str)
    )
    assert (
        ivy.default_float_dtype(input=None, float_dtype=None, as_native=False)
        == ivy.float32
    )
    assert ivy.default_float_dtype(float_dtype=ivy.float16) == ivy.float16
    assert ivy.default_float_dtype() == ivy.float32


# default_int_dtype
@handle_test(
    fn_tree="functional.ivy.default_int_dtype",
    dtype_x=helpers.dtype_and_values(available_dtypes=helpers.get_dtypes("integer")),
)
def test_default_int_dtype(
    *,
    dtype_x,
    native_array,
):
    int_dtype, x = dtype_x
    res = ivy.default_int_dtype(
        input=input, int_dtype=int_dtype[0], as_native=native_array
    )
    assert (
        isinstance(res, ivy.Dtype)
        or isinstance(res, typing.get_args(ivy.NativeDtype))
        or isinstance(res, ivy.NativeDtype)
        or isinstance(res, str)
    )
    assert (
        ivy.default_int_dtype(input=None, int_dtype=None, as_native=False) == ivy.int32
    )
    assert ivy.default_int_dtype(int_dtype=ivy.int16) == ivy.int16
    assert ivy.default_int_dtype() == ivy.int32


@st.composite
def dtypes_list(draw):
    num = draw(st.one_of(helpers.ints(min_value=1, max_value=5)))
    return draw(
        st.lists(
            st.sampled_from(ivy.valid_dtypes),
            min_size=num,
            max_size=num,
        )
    )


def _composition_1():
    return ivy.relu().argmax()


def _composition_2():
    a = ivy.floor
    return ivy.ceil() or a


# function_unsupported_dtypes
@handle_test(
    fn_tree="functional.ivy.function_supported_dtypes",
    func=st.sampled_from([_composition_1, _composition_2]),
    expected=helpers.get_dtypes("valid"),
)
def test_function_supported_dtypes(
    *,
    func,
    expected,
    ground_truth_backend,
):
    res = ivy.function_supported_dtypes(func)
    exp = set.intersection(set(expected), set(ivy.valid_dtypes))

    # Since float16 is only not valid for the torch backend, we remove from expected
    if "torch" in ivy.current_backend_str():
        exp.remove("float16")

    assert set(tuple(exp)) == set(res)


# function_unsupported_dtypes
@handle_test(
    fn_tree="functional.ivy.function_unsupported_dtypes",
    func=st.sampled_from([_composition_1, _composition_2]),
    expected=st.just([]),
)
def test_function_unsupported_dtypes(
    *,
    func,
    expected,
    ground_truth_backend,
):
    res = ivy.function_unsupported_dtypes(func)
    exp = set.union(set(expected), set(ivy.invalid_dtypes))

    # Since float16 is only not valid for the torch backend, we add to expected
    if "torch" in ivy.current_backend_str():
        exp.add("float16")

    assert set(tuple(exp)) == set(res)


# function_dtype_versioning
@handle_test(
    fn_tree="functional.ivy.function_unsupported_dtypes",  # dummy fn_tree
    func_and_version=st.just(
        [
            {
                "torch": {
                    "cumsum": {
                        "1.11.0": {"bfloat16", "uint8", "float16"},
                        "1.12.1": set(),
                    }
                }
            },
        ],
    ),
)
def test_function_dtype_versioning(
    *,
    func_and_version,
    backend_fw,
):
    for key in func_and_version:
        if key != backend_fw:
            continue
        var = ivy.get_backend().backend_version

        # key --> framework

        for key1 in func_and_version[key]:
            for key2 in func_and_version[key][key1]:
                var["version"] = key2
                fn = getattr(ivy.get_backend(), key1)
                expected = func_and_version[key][key1][key2]
                res = fn.unsupported_dtypes
                if res is None:
                    res = set()
                else:
                    res = set(res)
                if res != expected:
                    raise Exception
        return True


# function_dtype_versioning_frontend
@handle_test(
    fn_tree="functional.ivy.function_unsupported_dtypes",  # dummy fn_tree
    func_and_version=st.just(
        [
            {
                "torch": {
                    "cumsum": {
                        "1.11.0": {"bfloat16", "uint8", "float16"},
                        "1.12.1": set(),
                    }
                }
            },
        ],
    ),
)
def test_function_dtype_versioning_frontend(
    *,
    func_and_version,
    backend_fw,
):
    fw = backend_fw.current_backend_str()
    for key in func_and_version:
        if key != fw:
            continue
        frontend = importlib.import_module("ivy.functional.frontends")
        var = frontend.versions

        for key1 in func_and_version[key]:
            for key2 in func_and_version[key][key1]:
                var[fw] = key2
                fn = getattr(
                    importlib.import_module("ivy.functional.frontends." + fw), key1
                )
                expected = func_and_version[key][key1][key2]
                res = fn.unsupported_dtypes
                if res is None:
                    res = set()
                else:
                    res = set(res)
                if res != expected:
                    raise Exception
        return True


# invalid_dtype
@handle_test(
    fn_tree="functional.ivy.invalid_dtype",
    dtype_in=st.sampled_from(ivy.valid_dtypes),
)
def test_invalid_dtype(
    *,
    dtype_in,
    backend_fw,
):
    res = ivy.invalid_dtype(dtype_in)
    fw = backend_fw.current_backend_str()
    fw_invalid_dtypes = {
        "torch": ivy_torch.invalid_dtypes,
        "tensorflow": ivy_tf.invalid_dtypes,
        "jax": ivy_jax.invalid_dtypes,
        "numpy": ivy_np.invalid_dtypes,
    }
    if dtype_in in fw_invalid_dtypes[fw]:
        assert res is True, (
            f"fDtype = {dtype_in!r} is a valid dtype for {fw}, but" f"result = {res}"
        )
    else:
        assert res is False, (
            f"fDtype = {dtype_in!r} is not a valid dtype for {fw}, but"
            f"result = {res}"
        )


# unset_default_dtype
@handle_test(
    fn_tree="functional.ivy.unset_default_dtype",
    dtype=st.sampled_from(ivy.valid_dtypes),
)
def test_unset_default_dtype(
    *,
    dtype,
):
    stack_size_before = len(ivy.default_dtype_stack)
    ivy.set_default_dtype(dtype)
    ivy.unset_default_dtype()
    stack_size_after = len(ivy.default_dtype_stack)
    assert (
        stack_size_before == stack_size_after
    ), f"Default dtype not unset. Stack size= {stack_size_after!r}"


# unset_default_float_dtype
@handle_test(
    fn_tree="functional.ivy.unset_default_float_dtype",
    dtype=st.sampled_from(ivy.valid_float_dtypes),
)
def test_unset_default_float_dtype(
    *,
    dtype,
    with_out,
    backend_fw,
    fn_name,
    on_device,
):
    stack_size_before = len(ivy.default_float_dtype_stack)
    ivy.set_default_float_dtype(dtype)
    ivy.unset_default_float_dtype()
    stack_size_after = len(ivy.default_float_dtype_stack)
    assert (
        stack_size_before == stack_size_after
    ), f"Default float dtype not unset. Stack size= {stack_size_after!r}"


# unset_default_int_dtype
@handle_test(
    fn_tree="functional.ivy.unset_default_int_dtype",
    dtype=st.sampled_from(ivy.valid_int_dtypes),
)
def test_unset_default_int_dtype(
    *,
    dtype,
):
    stack_size_before = len(ivy.default_int_dtype_stack)
    ivy.set_default_int_dtype(dtype)
    ivy.unset_default_int_dtype()
    stack_size_after = len(ivy.default_int_dtype_stack)
    assert (
        stack_size_before == stack_size_after
    ), f"Default int dtype not unset. Stack size= {stack_size_after!r}"


# valid_dtype
@handle_test(
    fn_tree="functional.ivy.valid_dtype",
    dtype_in=st.sampled_from(ivy.valid_dtypes),
)
def test_valid_dtype(
    *,
    dtype_in,
    backend_fw,
):
    res = ivy.valid_dtype(dtype_in)
    fw = backend_fw.current_backend_str()
    fw_valid_dtypes = {
        "torch": ivy_torch.valid_dtypes,
        "tensorflow": ivy_tf.valid_dtypes,
        "jax": ivy_jax.valid_dtypes,
        "numpy": ivy_np.valid_dtypes,
    }
    if dtype_in in fw_valid_dtypes[fw]:
        assert res is True, (
            f"fDtype = {dtype_in!r} is not a valid dtype for {fw}, but"
            f"result = {res}"
        )
    else:
        assert res is False, (
            f"fDtype = {dtype_in!r} is a valid dtype for {fw}, but" f"result = {res}"
        )

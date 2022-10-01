# global
from hypothesis import assume, given, strategies as st
import math
import numpy as np

# local
from ivy.array import Array
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_cmd_line_args


_zero = np.asarray(0, dtype="uint8")
_one = np.asarray(1, dtype="uint8")


def _not_too_close_to_zero(x):
    f = np.vectorize(lambda item: item + (_one if np.isclose(item, 0) else _zero))
    return f(x)


@st.composite
def _pow_helper(draw, available_dtypes=None):
    if available_dtypes is None:
        available_dtypes = helpers.get_dtypes("numeric")
    dtype1, x1 = draw(
        helpers.dtype_and_values(
            available_dtypes=available_dtypes,
            small_abs_safety_factor=4,
            large_abs_safety_factor=4,
        )
    )
    dtype1 = dtype1[0]

    def cast_filter(dtype1_x1_dtype2):
        dtype1, _, dtype2 = dtype1_x1_dtype2
        if (ivy.as_ivy_dtype(dtype1), ivy.as_ivy_dtype(dtype2)) in ivy.promotion_table:
            return True
        return False

    dtype1, x1, dtype2 = draw(
        helpers.get_castable_dtype(draw(available_dtypes), dtype1, x1).filter(
            cast_filter
        )
    )
    if ivy.is_int_dtype(dtype2):
        max_val = ivy.iinfo(dtype2).max
    else:
        max_val = ivy.finfo(dtype2).max
    max_x1 = np.max(np.abs(x1[0]))
    if max_x1 in [0, 1]:
        max_value = None
    else:
        max_value = int(math.log(max_val) / math.log(max_x1))
        if abs(max_value) > abs(max_val) / 40 or max_value < 0:
            max_value = None
    dtype2, x2 = draw(
        helpers.dtype_and_values(
            small_abs_safety_factor=12,
            large_abs_safety_factor=12,
            safety_factor_scale="log",
            max_value=max_value,
            dtype=[dtype2],
        )
    )
    dtype2 = dtype2[0]
    if "int" in dtype2:
        x2 = ivy.nested_map(x2[0], lambda x: abs(x), include_derived={list: True})
    return [dtype1, dtype2], [x1, x2]


# __pos__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_array__pos__(
    dtype_and_x,
):
    _, x = dtype_and_x
    x_ = Array(x[0])
    ret = +x_
    np_ret = +x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __neg__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
    ),
)
def test_array__neg__(
    dtype_and_x,
):
    _, x = dtype_and_x
    x_ = Array(x[0])
    ret = -x_
    np_ret = -x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __pow__
@handle_cmd_line_args
@given(
    dtype_and_x=_pow_helper(),
)
def test_array__pow__(
    dtype_and_x,
):
    dtype, x = dtype_and_x
    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in dtype))
    # check if power isn't a float when x1 is integer
    assume(not (ivy.is_int_dtype(dtype[0]) and ivy.is_float_dtype(dtype[1])))
    # make power a non-negative data when both are integers
    if ivy.is_int_dtype(dtype[1]):
        x[1] = np.abs(x[1])
    x[0] = _not_too_close_to_zero(x[0])
    x[1] = _not_too_close_to_zero(x[1])
    data = Array(x[0])
    power = Array(x[1])
    ret = pow(data, power)
    np_ret = pow(x[0], x[1])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rpow__
@handle_cmd_line_args
@given(
    dtype_and_x=_pow_helper(),
)
def test_array__rpow__(
    dtype_and_x,
):
    dtype, x = dtype_and_x
    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in dtype))
    # check if power isn't a float when x1 is integer
    assume(not (ivy.is_int_dtype(dtype[0]) and ivy.is_float_dtype(dtype[1])))
    # make power a non-negative data when both are integers
    if ivy.is_int_dtype(dtype[1]) and ivy.is_int_dtype(dtype[0]):
        x[1] = np.abs(x[1])

    x[0] = _not_too_close_to_zero(x[0])
    x[1] = _not_too_close_to_zero(x[1])
    data = Array(x[1])
    power = Array(x[0])
    ret = data.__rpow__(power)
    np_ret = x[1].__rpow__(x[0])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __ipow__
@handle_cmd_line_args
@given(
    dtype_and_x=_pow_helper(),
)
def test_array__ipow__(
    dtype_and_x,
):
    dtype, x = dtype_and_x
    # bfloat16 is not supported by numpy
    assume(not ("bfloat16" in dtype))
    # check if power isn't a float when data is integer
    assume(not (ivy.is_int_dtype(dtype[0]) and ivy.is_float_dtype(dtype[1])))
    # make power a non-negative data when both are integers
    if ivy.is_int_dtype(dtype[1]) and ivy.is_int_dtype(dtype[0]):
        x[1] = np.abs(x[1])

    x[0] = _not_too_close_to_zero(x[0])
    x[1] = _not_too_close_to_zero(x[1])
    data = Array(x[0])
    power = Array(x[1])
    ret = data.__ipow__(power)
    np_ret = pow(x[0], x[1])
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __add__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__add__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data + other
    np_ret = x[0] + x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __radd__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__radd__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__radd__(other)
    np_ret = x[0] + x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __iadd__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__iadd__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__iadd__(other)
    np_ret = x[0] + x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __sub__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__sub__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data - other
    np_ret = x[0] - x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __rsub__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__rsub__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__rsub__(other)
    np_ret = x[1] - x[0]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )


# __isub__
@handle_cmd_line_args
@given(
    dtype_and_x=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("numeric"),
        num_arrays=2,
        large_abs_safety_factor=2.5,
        small_abs_safety_factor=2.5,
        safety_factor_scale="log",
    ),
)
def test_array__isub__(
    dtype_and_x,
):
    _, x = dtype_and_x
    data = Array(x[0])
    other = Array(x[1])
    ret = data.__isub__(other)
    np_ret = x[0] - x[1]
    ret = helpers.flatten_and_to_np(ret=ret)
    ret_gt = helpers.flatten_and_to_np(ret=np_ret)
    for (_, _) in zip(ret, ret_gt):
        helpers.value_test(
            ret=ret,
            ret_from_gt=ret_gt,
            ground_truth_backend="numpy",
        )

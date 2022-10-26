"""Collection of TensorFlow gradient functions, wrapped to fit Ivy syntax and
signature.
"""

# global
import tensorflow as tf
from typing import Union, Optional, Callable
import numpy as np

# local
import ivy
from ivy.functional.ivy.gradients import (
    _arrays_to_float_variables,
    _get_required_native_variables,
    _get_native_variables_and_indices,
    _remove_zeros_and_nones,
    _stop_grad_and_index,
)


def variable(x):
    with tf.device(ivy.dev(x, as_native=True)):
        return tf.Variable(x, trainable=True)


def is_variable(x, exclusive=False):
    return isinstance(x, tf.Variable)


def variable_data(x):
    return x.value()


def execute_with_gradients(
    func, xs, /, *, retain_grads=False, xs_grad_idxs=None, ret_grad_idxs=None
):
    xs = _arrays_to_float_variables(xs)
    with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
        tape.watch(ivy.to_native(xs))
        func_ret = func(xs)
    xs = _get_required_native_variables(xs, xs_grad_idxs)
    ret_idxs, ret_values = _get_native_variables_and_indices(func_ret, reshape=False)
    if ret_values is None or (isinstance(ret_values, list) and len(ret_values) == 0):
        return func_ret, {}
    if isinstance(ret_values, list) and len(ret_values) == 1:
        y = ret_values[0]
    else:
        y = ret_values
    grad_func = lambda y: tape.gradient(y, ivy.to_native(xs))
    if isinstance(y, ivy.NativeArray):
        grads = ivy.to_ivy(grad_func(y))
    else:
        array_idxs = ivy.nested_argwhere(y, lambda x: ivy.is_native_array(x))
        if (
            not isinstance(array_idxs, list)
            or np.asarray(array_idxs, "object").size == 0
        ):
            y = []
        else:
            y = ivy.multi_index_nest(y, array_idxs)
        grads_ = ivy.nested_map(y, grad_func, include_derived=True)
        grads = grads_
        if isinstance(ret_idxs, list) and len(ret_idxs):
            grads = {ret_idxs[i]: grad for i, grad in enumerate(grads_)}
    grads = ivy.nested_map(
        grads, lambda x: ivy.where(ivy.isnan(x), 0, x) if ivy.is_ivy_array(x) else x
    )
    grads = _remove_zeros_and_nones(grads, grads)
    func_ret, grads = _stop_grad_and_index(func_ret, retain_grads, grads, ret_grad_idxs)
    if not retain_grads:
        del tape
    grads = ivy.to_ivy(grads)
    return func_ret, grads


def value_and_grad(func):
    def grad_fn(xs):
        grads = ivy.nested_map(xs, lambda x: ivy.zeros_like(x), include_derived=True)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            xs = ivy.nested_map(xs, lambda x: ivy.to_native(x), include_derived=True)
            tape.watch(xs)
            y = func(xs)
        y = y.to_native(y)
        grads_ = tape.gradient(y, xs)
        grads_ = ivy.nested_map(
            grads_,
            lambda x: ivy.to_ivy(x),
            include_derived=True,
        )
        grads_ = _remove_zeros_and_nones(grads_, grads_)
        grads_ = ivy.to_ivy(grads_)
        grad_idxs = ivy.nested_argwhere(grads_, lambda x: ivy.is_ivy_array(x))
        grad_array_vals = list(ivy.multi_index_nest(grads_, grad_idxs))
        xs = ivy.to_ivy(xs)
        if isinstance(xs, ivy.Array):
            grads = grads_
        else:
            ivy.set_nest_at_indices(grads, grad_idxs, grad_array_vals)
        y = ivy.to_ivy(y)
        return y, grads

    return grad_fn


def stop_gradient(
    x: Union[tf.Tensor, tf.Variable],
    preserve_type: bool = True,
    *,
    out: Optional[Union[tf.Tensor, tf.Variable]] = None,
) -> Union[tf.Tensor, tf.Variable]:
    is_var = is_variable(x)
    x = tf.stop_gradient(x)
    if is_var and preserve_type:
        return variable(x)
    return x


def jac(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))

    def callback_fn(x_in):
        with tf.GradientTape() as tape:
            x_in = ivy.to_native(x_in)
            tape.watch(x_in)
            y = grad_fn(x_in)
        return ivy.to_ivy(tape.jacobian(y, x_in))

    return callback_fn


def grad(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))

    def callback_fn(x_in):
        with tf.GradientTape() as tape:
            x_in = ivy.to_native(ivy.array(x_in))
            tape.watch(x_in)
            y = grad_fn(x_in)
        grad_ = ivy.to_ivy(tape.gradient(y, x_in))
        return _remove_zeros_and_nones(grad_, grad_)

    return callback_fn

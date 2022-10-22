"""Collection of Jax gradient functions, wrapped to fit Ivy syntax and signature."""

# global
import jax
import jax.lax as jlax
import jaxlib
from jaxlib.xla_extension import Buffer
from ivy.functional.backends.jax import JaxArray
from typing import Optional, Callable
from itertools import chain


# local
import ivy
from ivy.functional.ivy.gradients import (
    _arrays_to_float_variables,
    _get_required_native_variables,
    _get_native_variables_and_indices,
    _remove_zeros_and_nones,
    _stop_grad_and_index,
)


# ToDo: modify these functions to track whether variable() has been called
def variable(x):
    return x


def is_variable(x, exclusive=False):
    if exclusive:
        return False
    return isinstance(
        x, (jax.interpreters.xla._DeviceArray, jaxlib.xla_extension.DeviceArray, Buffer)
    )


def variable_data(x):
    return x


def _set_duplicates(xs, duplicate_key_chains):
    originals = [
        [key_chains[0]] * (len(key_chains) - 1) for key_chains in duplicate_key_chains
    ]
    originals = ivy.multi_index_nest(xs, list(chain(*originals)))
    duplicates = [list(key_chains[1:]) for key_chains in duplicate_key_chains]
    nullifying_key_chains = [
        keychains.split("/") for keychains in list(chain(*duplicates))
    ]
    ivy.set_nest_at_indices(xs, nullifying_key_chains, originals)
    return xs


def _forward_fn(xs, x, xs_grad_idxs, func, duplicate_key_chains):
    if xs_grad_idxs is not None:
        ivy.set_nest_at_indices(xs, xs_grad_idxs, x)
    else:
        xs = x
    if isinstance(xs, ivy.Container):
        xs = _set_duplicates(xs, duplicate_key_chains)
    ret = func(xs)
    _, ret_values = _get_native_variables_and_indices(ret)
    if isinstance(ret_values, list) and len(ret_values) == 1:
        ret_values = ret_values[0]
    return ret_values


def execute_with_gradients(
    func, xs, /, *, retain_grads=False, xs_grad_idxs=None, ret_grad_idxs=None
):
    xs = _arrays_to_float_variables(xs)
    xs = ivy.stop_gradient(xs)
    func_ret = func(xs)
    xs_required = _get_required_native_variables(ivy.copy_nest(xs), xs_grad_idxs)
    xs = ivy.to_native(xs)
    ret_idxs, ret_values = _get_native_variables_and_indices(func_ret)
    if ret_values is None or (isinstance(ret_values, list) and len(ret_values) == 0):
        return func_ret, {}
    if isinstance(ret_values, list) and len(ret_values) == 1:
        y = ret_values[0]
    else:
        y = ret_values
    duplicate_key_chains = ()
    if isinstance(xs, ivy.Container):
        duplicate_key_chains = xs.duplicate_array_keychains()
    if isinstance(y, ivy.NativeArray):
        grad_fn = jax.grad(
            lambda x: _forward_fn(xs, x, xs_grad_idxs, func, duplicate_key_chains)
        )
        grads = grad_fn(xs_required)
    else:
        grad_fn = jax.jacrev(
            lambda x: _forward_fn(xs, x, xs_grad_idxs, func, duplicate_key_chains)
        )
        grads_ = grad_fn(xs_required)
        grads = grads_
        if isinstance(ret_idxs, list) and len(ret_idxs):
            grads = {ret_idxs[i]: grad for i, grad in enumerate(grads_)}
    if isinstance(xs, ivy.Container):
        grads = _set_duplicates(grads, duplicate_key_chains)
    grads = _remove_zeros_and_nones(grads, grads)
    func_ret, grads = _stop_grad_and_index(func_ret, retain_grads, grads, ret_grad_idxs)
    grads = ivy.to_ivy(grads)
    return func_ret, grads


def value_and_grad(func):
    grad_fn = lambda xs: ivy.to_native(func(xs))

    def callback_fn(xs):
        xs = ivy.nested_map(xs, lambda x: ivy.to_native(x), include_derived=True)
        value, grad = jax.value_and_grad(grad_fn)(xs)
        grad = _remove_zeros_and_nones(grad, grad)
        return ivy.to_ivy(value), ivy.to_ivy(grad)

    return callback_fn


def stop_gradient(
    x: JaxArray, preserve_type: bool = True, *, out: Optional[JaxArray] = None
) -> JaxArray:
    return jlax.stop_gradient(x)


def jac(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))
    callback_fn = lambda x_in: ivy.to_ivy(jax.jacfwd(grad_fn)((ivy.to_native(x_in))))
    return callback_fn


def grad(func: Callable):
    grad_fn = lambda x_in: ivy.to_native(func(x_in))
    callback_fn = lambda x_in: ivy.to_ivy(jax.grad(grad_fn)(ivy.to_native(x_in)))
    return callback_fn

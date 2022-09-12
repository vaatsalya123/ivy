import ivy
from ivy.functional.ivy.extensions import (
    _verify_coo_components,
    _is_data_not_indices_values_and_shape,
    _is_coo_not_csr,
)
import tensorflow as tf
import logging


def is_native_sparse_array(x):
    return isinstance(x, tf.SparseTensor)


def native_sparse_array(
    data=None,
    *,
    coo_indices=None,
    csr_crow_indices=None,
    csr_col_indices=None,
    values=None,
    dense_shape=None
):
    if _is_data_not_indices_values_and_shape(
        data, coo_indices, csr_crow_indices, csr_col_indices, values, dense_shape
    ):
        assert ivy.is_native_sparse_array(data), "not a sparse array"
        return data
    elif _is_coo_not_csr(
        coo_indices, csr_crow_indices, csr_col_indices, values, dense_shape
    ):
        _verify_coo_components(
            indices=coo_indices, values=values, dense_shape=dense_shape
        )
        return tf.SparseTensor(
            indices=coo_indices, values=values, dense_shape=dense_shape
        )
    else:
        logging.warning(
            "Tensorflow does not support CSR sparse array natively. None is returned."
        )
        return None


def native_sparse_array_to_indices_values_and_shape(x):
    if isinstance(x, tf.SparseTensor):
        return x.indices, x.values, x.dense_shape
    raise Exception("not a SparseTensor")

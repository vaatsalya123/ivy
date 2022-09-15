from typing import Optional, Tuple

import ivy
import numpy as np


# Array API Standard #
# ------------------ #


def argmax(
    x: np.ndarray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.argmax(x, axis=axis, keepdims=keepdims, out=out)
    return np.array(ret)


argmax.support_native_out = True


def argmin(
    x: np.ndarray,
    /,
    *,
    axis: Optional[int] = None,
    keepdims: bool = False,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    ret = np.argmin(x, axis=axis, keepdims=keepdims, out=out)
    return np.array(ret)


argmin.support_native_out = True


def nonzero(
    x: np.ndarray,
    /,
) -> Tuple[np.ndarray]:
    return np.nonzero(x)


def where(
    condition: np.ndarray,
    x1: np.ndarray,
    x2: np.ndarray,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    x1, x2 = ivy.promote_types_of_inputs(x1, x2)
    return np.where(condition, x1, x2)


# Extra #
# ----- #


def indices_where(x: np.ndarray, /, *, out: Optional[np.ndarray] = None) -> np.ndarray:
    where_x = np.where(x)
    if len(where_x) == 1:
        return np.expand_dims(where_x[0], -1)
    res = np.concatenate([np.expand_dims(item, -1) for item in where_x], -1, out=out)
    return res


indices_where.support_native_out = True

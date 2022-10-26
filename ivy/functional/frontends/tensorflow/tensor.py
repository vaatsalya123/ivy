# global

# local
import ivy
import ivy.functional.frontends.tensorflow as tf_frontend


class Tensor:
    def __init__(self, data):
        if ivy.is_native_array(data):
            data = ivy.Array(data)
        elif isinstance(data, list):
            data = ivy.asarray(data)
        self.data = data

    def __repr__(self):
        return (
            "ivy.functional.frontends.tensorflow.tensor("
            + str(ivy.to_list(self.data))
            + ")"
        )

    # Instance Methods #
    # -------------------#

    def get_shape(self):
        return tf_frontend.raw_ops.Shape(input=self.data)

    def __add__(self, y, name="add"):
        return y.__radd__(self.data)

    def __div__(self, x, name="div"):
        return tf_frontend.math.divide(x, self.data, name=name)

    def __and__(self, y, name="and"):
        return y.__rand__(self.data)

    def __array__(self, dtype=None, name="array"):
        return ivy.asarray(self.data, dtype=dtype)

    def __bool__(self, name="bool"):
        if isinstance(self.data, int):
            return self.data != 0

        temp = ivy.squeeze(ivy.asarray(self.data), axis=None)
        shape = ivy.shape(temp)
        if shape:
            raise ivy.exceptions.IvyError(
                "The truth value of an array with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )

        return temp != 0

    def __eq__(self, other):
        return tf_frontend.raw_ops.Equal(
            x=self.data, y=other, incompatible_shape_error=False
        )

    def __floordiv__(self, y, name="floordiv"):
        return y.__rfloordiv__(self.data)

    def __ge__(self, y, name="ge"):
        return tf_frontend.raw_ops.GreaterEqual(x=self.data, y=y.data, name=name)

    def __getitem__(self, slice_spec, var=None, name="getitem"):
        return Tensor(self.data.__getitem__(slice_spec))

    def __gt__(self, y, name="gt"):
        return tf_frontend.raw_ops.Greater(x=self.data, y=y.data, name=name)

    def __invert__(self, name="invert"):
        return tf_frontend.raw_ops.Invert(x=self.data, name=name)

    def __le__(self, y, name="le"):
        return tf_frontend.raw_ops.LessEqual(x=self.data, y=y.data, name=name)

    def __lt__(self, y, name="lt"):
        return tf_frontend.raw_ops.Less(x=self.data, y=y.data, name=name)

    def __matmul__(self, y, name="matmul"):
        return y.__rmatmul__(self.data)

    def __mul__(self, x, name="mul"):
        return tf_frontend.math.multiply(x, self.data, name=name)

    def __ne__(self, other):
        return tf_frontend.raw_ops.NotEqual(
            x=self.data, y=other.data, incompatible_shape_error=False
        )

    def __neg__(self, name="neg"):
        return tf_frontend.raw_ops.Neg(x=self.data, name=name)

    __nonzero__ = __bool__

    def __or__(self, y, name="or"):
        return y.__ror__(self.data)

    def __radd__(self, x, name="radd"):
        return tf_frontend.math.add(x, self.data, name=name)

    def __rand__(self, x, name="rand"):
        return tf_frontend.math.logical_and(x, self.data, name=name)

    def __rfloordiv__(self, x, name="rfloordiv"):
        return tf_frontend.raw_ops.FloorDiv(x=x, y=self.data, name=name)

    def __rmatmul__(self, x, name="rmatmul"):
        return tf_frontend.raw_ops.MatMul(a=x, b=self.data, name=name)

    def __rmul__(self, x, name="rmul"):
        return tf_frontend.raw_ops.Mul(x=x, y=self.data, name=name)

    def __ror__(self, x, name="ror"):
        return tf_frontend.raw_ops.LogicalOr(x=x, y=self.data, name=name)

    def __rsub__(self, x, name="rsub"):
        return tf_frontend.math.subtract(x, self.data, name=name)

    def __rtruediv__(self, x, name="rtruediv"):
        return tf_frontend.math.truediv(x, self.data, name=name)

    def __rxor__(self, x, name="rxor"):
        return tf_frontend.math.logical_xor(x, self.data, name=name)

    def __sub__(self, y, name="sub"):
        return y.__rsub__(self.data)

    def __truediv__(self, y, name="truediv"):
        dtype = ivy.dtype(self.data)
        if dtype in [ivy.uint8, ivy.int8, ivy.uint16, ivy.int16]:
            return ivy.astype(y, ivy.float32).__rtruediv__(
                ivy.astype(self.data, ivy.float32)
            )
        if dtype in [ivy.uint32, ivy.int32, ivy.uint64, ivy.int64]:
            return ivy.astype(y, ivy.float64).__rtruediv__(
                ivy.astype(self.data, ivy.float64)
            )
        return y.__rtruediv__(self.data)

    def __len__(self):
        raise ivy.exceptions.IvyError(
            "len is not well defined for a symbolic Tensor. Please call `x.shape` "
            "rather than `len(x)` for shape information. "
        )

    def __xor__(self, y, name="xor"):
        return y.__rxor__(self.data)

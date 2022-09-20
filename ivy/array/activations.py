# global
import abc
from typing import Optional, Union

# local
import ivy


# ToDo: implement all methods here as public instance methods


class ArrayWithActivations(abc.ABC):
    def relu(self: ivy.Array, /, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.relu. This method simply wraps the
        function, and so the docstring for ivy.relu also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([-1., 0., 1.])
        >>> y = x.relu()
        >>> print(y)
        ivy.array([0., 0., 1.])
        """
        return ivy.relu(self._data, out=out)

    def leaky_relu(
        self: ivy.Array,
        /,
        *,
        alpha: float = 0.2,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.leaky_relu. This method simply wraps
        the function, and so the docstring for ivy.leaky_relu also applies to this
        method with minimal changes.

        Examples
        --------
        >>> x = ivy.array([0.39, -0.85])
        >>> y = x.leaky_relu()
        >>> print(y)
        ivy.array([ 0.39, -0.17])
        """
        return ivy.leaky_relu(self._data, alpha=alpha, out=out)

    def gelu(
        self: ivy.Array,
        /,
        *,
        approximate: bool = True,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.gelu. This method simply wraps the
        function, and so the docstring for ivy.gelu also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([0.3, -0.1])
        >>> y = x.gelu()
        >>> print(y)
        ivy.array([ 0.185, -0.046])
        """
        return ivy.gelu(self._data, approximate=approximate, out=out)

    def sigmoid(self: ivy.Array, /, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sigmoid. This method simply wraps the
        function, and so the docstring for ivy.sigmoid also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([-1., 1., 2.])
        >>> y = x.sigmoid()
        >>> print(y)
        ivy.array([0.269, 0.731, 0.881])
        """
        return ivy.sigmoid(self._data, out=out)

    def softmax(
        self: ivy.Array,
        /,
        *,
        axis: Optional[int] = None,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.softmax. This method simply wraps the
        function, and so the docstring for ivy.softmax also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([1.0, 0, 1.0])
        >>> y = x.softmax()
        >>> print(y)
        ivy.array([0.422, 0.155, 0.422])
        """
        return ivy.softmax(self._data, axis=axis, out=out)

    def softplus(self: ivy.Array,
                 /,
                 *,
                 beta: Optional[Union[int, float]] = None,
                 threshold: Optional[Union[int, float]] = None,
                 out: Optional[ivy.Array] = None
                 ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.softplus. This method simply wraps the
        function, and so the docstring for ivy.softplus also applies to this method
        with minimal changes.

        Examples
        --------
        >>> x = ivy.array([-0.3461, -0.6491])
        >>> y = x.softplus()
        >>> print(y)
        ivy.array([0.535, 0.42 ])

        >>> x = ivy.array([-0.3461, -0.6491])
        >>> x.softplus(beta=0.5)
        >>> print(y)
        ivy.array([1.22, 1.09])

        >>> ivy.array([1.31, 2., 2.])
        >>> x.softplus(threshold=2)
        >>> print(y)
        ivy.array([2.15, 2.63, 2.63])

        """
        return ivy.softplus(self._data, beta=beta, threshold=threshold, out=out)

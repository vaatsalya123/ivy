# global
import abc
from typing import Optional

# local
import ivy


class ArrayWithExtensions(abc.ABC):
    def sinc(self: ivy.Array, *, out: Optional[ivy.Array] = None) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.sinc. This method simply wraps the
        function, and so the docstring for ivy.sinc also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input array whose elements are each expressed in radians. Should have a
            floating-point data type.
        out
            optional output array, for writing the result to. It must have a shape that
            the inputs broadcast to.

        Returns
        -------
        ret
            an array containing the sinc of each element in ``self``. The returned
            array must have a floating-point data type determined by
            :ref:`type-promotion`.

        Examples
        --------
        >>> x = ivy.array([0.5, 1.5, 2.5, 3.5])
        >>> y = x.sinc()
        >>> print(y)
        ivy.array([0.637,-0.212,0.127,-0.0909])
        """
        return ivy.sinc(self._data, out=out)

    def flatten(
        self: ivy.Array,
        *,
        start_dim: int,
        end_dim: int,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        """ivy.Array instance method variant of ivy.flatten. This method simply
        wraps the function, and so the docstring for ivy.unstack also applies to
        this method with minimal changes.

        Parameters
        ----------
        self
            input array to flatten. 
        start_dim
            first dim to flatten. If not set, defaults to 0.
        end_dim
            last dim to flatten. If not set, defaults to -1.

        Returns
        -------
        ret
            the flattened array over the specified dimensions. 

        Examples
        --------
        >>> x = ivy.array([1,2], [3,4])
        >>> ivy.flatten(x)
        ivy.array([1, 2, 3, 4])

        >>> x = ivy.array(
            [[[[ 5,  5,  0,  6],
            [17, 15, 11, 16],
            [ 6,  3, 13, 12]],

            [[ 6, 18, 10,  4],
            [ 5,  1, 17,  3],
            [14, 14, 18,  6]]],


        [[[12,  0,  1, 13],
            [ 8,  7,  0,  3],
            [19, 12,  6, 17]],

            [[ 4, 15,  6, 15],
            [ 0,  5, 17,  9],
            [ 9,  3,  6, 19]]],


        [[[17, 13, 11, 16],
            [ 4, 18, 17,  4],
            [10, 10,  9,  1]],

            [[19, 17, 13, 10],
            [ 4, 19, 16, 17],
            [ 2, 12,  8, 14]]]]
            )
        >>> ivy.flatten(x, start_dim = 1, end_dim = 2)
        ivy.array(
            [[[ 5,  5,  0,  6],
            [17, 15, 11, 16],
            [ 6,  3, 13, 12],
            [ 6, 18, 10,  4],
            [ 5,  1, 17,  3],
            [14, 14, 18,  6]],

            [[12,  0,  1, 13],
            [ 8,  7,  0,  3],
            [19, 12,  6, 17],
            [ 4, 15,  6, 15],
            [ 0,  5, 17,  9],
            [ 9,  3,  6, 19]],

            [[17, 13, 11, 16],
            [ 4, 18, 17,  4],
            [10, 10,  9,  1],
            [19, 17, 13, 10],
            [ 4, 19, 16, 17],
            [ 2, 12,  8, 14]]]))
        """
        return ivy.flatten(
            self._data, 
            start_dim=start_dim, 
            end_dim=end_dim, 
            out=out)
        
    def lcm(
        self: ivy.Array,
        x2: ivy.Array,
        *,
        out: Optional[ivy.Array] = None
    ) -> ivy.Array:
        """
        ivy.Array instance method variant of ivy.lcm. This method simply wraps the
        function, and so the docstring for ivy.lcm also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            first input array.
        x2
            second input array
        out
            optional output array, for writing the result to. 

        Returns
        -------
        ret
            an array that includes the element-wise least common multiples
            of 'self' and x2

        Examples
        --------
        >>> x1=ivy.array([2, 3, 4])
        >>> x2=ivy.array([5, 8, 15])
        >>> x1.lcm(x2)
        ivy.array([10, 21, 60])
        """
        return ivy.lcm(self, x2, out=out)

# global
from typing import Optional, Tuple, Union, List

# local
from ivy.container.base import ContainerBase
import ivy

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithLayers(ContainerBase):
    @staticmethod
    def static_depthwise_conv2d(
        x: ivy.Container,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: Union[str, List[int]],
        data_format: str = "NHWC",
        dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.depthwise_conv2d. This method simply
        wraps the function, and so the docstring for ivy.depthwise_conv2d also applies
        to this method with minimal changes.

        Parameters
        ----------
        x
            Input image *[batch_size,h,w,d]*.
        filters
            Convolution filters *[fh,fw,d_in]*. (d_in must be the same as d from x)
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating the
            per-dimension paddings.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the convolution operation.

        Examples
        --------
        >>> a = ivy.randint(0, 255, (1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> b = ivy.randint(0, 255, (1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> inp = ivy.Container(a=a, b=b)
        >>> filters = ivy.random_normal(0, 1, [3, 3, 3])
        >>> y = ivy.Container.static_depthwise_conv2d( \
                                                    inp, \
                                                    filters, \
                                                    strides=2, \
                                                    padding='SAME')
        >>> print(y.shape)
        [1, 64, 64, 3]
        """
        return ContainerBase.multi_map_in_static_method(
            "depthwise_conv2d",
            x,
            filters=filters,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilations=dilations,
            out=out,
        )

    def depthwise_conv2d(
        self: ivy.Container,
        filters: Union[ivy.Array, ivy.NativeArray],
        strides: Union[int, Tuple[int], Tuple[int, int]],
        padding: Union[str, List[int]],
        data_format: str = "NHWC",
        dilations: Optional[Union[int, Tuple[int], Tuple[int, int]]] = 1,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.depthwise_conv2d. This method
        simply wraps the function, and so the docstring for ivy.depthwise_conv2d
        also applies to this method with minimal changes.

        Parameters
        ----------
        self
            Input image *[batch_size,h,w,d]*.
        filters
            Convolution filters *[fh,fw,d_in]*. (d_in must be the same as d from self)
        strides
            The stride of the sliding window for each dimension of input.
        padding
            "SAME" or "VALID" indicating the algorithm, or list indicating the
            per-dimension paddings.
        data_format
            "NHWC" or "NCHW". Defaults to "NHWC".
        dilations
            The dilation factor for each dimension of input. (Default value = 1)
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            The result of the convolution operation.

        Examples
        --------
        >>> a = ivy.randint(0, 255, (1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> b = ivy.randint(0, 255, (1, 128, 128, 3)).astype(ivy.float32) / 255.0
        >>> inp = ivy.Container(a=a, b=b)
        >>> filters = ivy.random_normal(0, 1, [3, 3, 3])
        >>> y = inp.depthwise_conv2d(filters, strides=2, padding='SAME')
        >>> print(y.shape)
        [1, 64, 64, 3]
        """
        return self.static_depthwise_conv2d(
            self, filters, strides, padding, data_format, dilations, out=out
        )

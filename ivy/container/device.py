# local
from typing import Union, Optional, Any, List, Dict

import ivy
from ivy.container.base import ContainerBase


# ToDo: implement all methods here as public instance methods

# noinspection PyMissingConstructor
class ContainerWithDevice(ContainerBase):
    @staticmethod
    def static_dev(x: ivy.Container, as_native: bool = False) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.dev. This method simply
        wraps the function, and so the docstring for ivy.dev also applies to this
        method with minimal changes.

        """
        return ContainerBase.multi_map_in_static_method("dev", x, as_native=as_native)

    def dev(self: ivy.Container, as_native: bool = False) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.dev. This method simply
        wraps the function, and so the docstring for ivy.dev also applies to this
        method with minimal changes.

        """
        return self.static_dev(self, as_native=as_native)

    @staticmethod
    def static_to_device(
        x: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        device: Union[ivy.Device, ivy.NativeDevice],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        stream: Optional[Union[int, Any]] = None,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.to_device. This method
        simply wraps the function, and so the docstring for ivy.to_device also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
           input array to be moved to the desired device
        device
            device to move the input array `x` to
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        stream
            stream object to use during copy. In addition to the types supported
            in array.__dlpack__(), implementations may choose to support any
            library-specific stream object with the caveat that any code using
            such an object would not be portable.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            input array x placed on the desired device

        """
        return ContainerBase.multi_map_in_static_method(
            "to_device",
            x,
            device,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            stream=stream,
            out=out,
        )

    def to_device(
        self: ivy.Container,
        device: Union[ivy.Device, ivy.NativeDevice],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        stream: Optional[Union[int, Any]] = None,
        out: Optional[ivy.Container] = None
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.to_device. This method
        simply wraps the function, and so the docstring for ivy.to_device also
        applies to this method with minimal changes.

        Parameters
        ----------
        x
           input array to be moved to the desired device
        device
            device to move the input array `x` to
        key_chains
            The key-chains to apply or not apply the method to. Default is None.
        to_apply
            If True, the method will be applied to key_chains, otherwise key_chains
            will be skipped. Default is True.
        prune_unapplied
            Whether to prune key_chains for which the function was not applied.
            Default is False.
        map_sequences
            Whether to also map method to sequences (lists, tuples). Default is False.
        stream
            stream object to use during copy. In addition to the types supported
            in array.__dlpack__(), implementations may choose to support any
            library-specific stream object with the caveat that any code using
            such an object would not be portable.
        out
            optional output array, for writing the result to. It must have a
            shape that the inputs broadcast to.

        Returns
        -------
        ret
            input array x placed on the desired device

        """
        return self.static_to_device(
            self,
            device,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            stream=stream,
            out=out,
        )

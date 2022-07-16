# global
from typing import Optional, Union, List, Tuple, Dict, Iterable
from numbers import Number

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithManipulation(ContainerBase):
    def concat(
        self: ivy.Container,
        xs: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        axis: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_nests: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        conts = [self]
        arrays = [None]
        for x in xs:
            if ivy.is_ivy_container(x):
                conts.append(x)
                arrays.append(None)
            else:
                arrays.append(x)
        return ContainerBase.handle_inplace(
            ContainerBase.multi_map(
                lambda xs_, _: ivy.concat(
                    xs=[a if ivy.exists(a) else xs_.pop(0) for a in arrays], axis=axis
                )
                if ivy.is_array(xs_[0])
                else xs_,
                conts,
                key_chains,
                to_apply,
                prune_unapplied,
                map_nests=map_nests,
            ),
            out=out,
        )

    def expand_dims(
        self: ivy.Container,
        axis: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.expand_dims(x_, axis=axis)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def permute_dims(
        self: ivy.Container,
        axes: Tuple[int, ...],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.permute_dims(x_, axes=axes)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def flip(
        self: ivy.Container,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.flip(x_, axis=axis) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def reshape(
        self: ivy.Container,
        shape: Union[ivy.Shape, ivy.NativeShape, ivy.Container],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.reshape(x_, shape=shape) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    @staticmethod
    def static_roll(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        shift: Union[int, Tuple[int, ...], ivy.Container],
        axis: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.roll. This method simply wraps the
        function, and so the docstring for ivy.roll also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            input container.
        shift
            number of places by which the elements are shifted. If ``shift`` is a tuple,
            then ``axis`` must be a tuple of the same size, and each of the given axes
            must be shifted by the corresponding element in ``shift``. If ``shift`` is
            an ``int`` and ``axis`` a tuple, then the same ``shift`` must be used for
            all specified axes. If a shift is positive, then array elements must be
            shifted positively (toward larger indices) along the dimension of ``axis``.
            If a shift is negative, then array elements must be shifted negatively
            (toward smaller indices) along the dimension of ``axis``.
        axis
            axis (or axes) along which elements to shift. If ``axis`` is ``None``, the
            array must be flattened, shifted, and then restored to its original shape.
            Default ``None``.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container having the same data type as ``x`` and whose elements,
            relative to ``x``, are shifted.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                              b=ivy.array([3., 4., 5.]))
        >>> y = ivy.Container.static_roll(x, 1)
        >>> print(y)
        {
            a: ivy.array([2., 0., 1.]),
            b: ivy.array([5., 3., 4.])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                              b=ivy.array([3., 4., 5.]))
        >>> shift = ivy.Container(a=1, b=-1)
        >>> y = ivy.Container.static_roll(x, shift)
        >>> print(y)
        {
            a: ivy.array([2., 0., 1.]),
            b: ivy.array([4., 5., 3.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "roll",
            x,
            shift,
            axis,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def roll(
        self: ivy.Container,
        shift: Union[int, Tuple[int, ...], ivy.Container],
        axis: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.roll. This method simply wraps the
        function, and so the docstring for ivy.roll also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            input container.
        shift
            number of places by which the elements are shifted. If ``shift`` is a tuple,
            then ``axis`` must be a tuple of the same size, and each of the given axes
            must be shifted by the corresponding element in ``shift``. If ``shift`` is
            an ``int`` and ``axis`` a tuple, then the same ``shift`` must be used for
            all specified axes. If a shift is positive, then array elements must be
            shifted positively (toward larger indices) along the dimension of ``axis``.
            If a shift is negative, then array elements must be shifted negatively
            (toward smaller indices) along the dimension of ``axis``.
        axis
            axis (or axes) along which elements to shift. If ``axis`` is ``None``, the
            array must be flattened, shifted, and then restored to its original shape.
            Default ``None``.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            an output container having the same data type as ``self`` and whose
            elements, relative to ``self``, are shifted.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = x.roll(1)
        >>> print(y)
        {
            a: ivy.array([2., 0., 1.]),
            b: ivy.array([5., 3., 4.])
        }
        """
        return self.static_roll(
            self,
            shift,
            axis,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    def squeeze(
        self: ivy.Container,
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.squeeze(x_, axis=axis) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def stack(
        self: ivy.Container,
        x: Union[
            Tuple[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
            List[Union[ivy.Array, ivy.NativeArray, ivy.Container]],
        ],
        axis: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_nests: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        conts = [self]
        arrays = [None]
        for y in x:
            if ivy.is_ivy_container(y):
                conts.append(y)
                arrays.append(None)
            else:
                arrays.append(y)
        return ContainerBase.handle_inplace(
            ContainerBase.multi_map(
                lambda xs_, _: ivy.stack(
                    x=[a if ivy.exists(a) else xs_.pop(0) for a in arrays], axis=axis
                )
                if ivy.is_array(xs_[0])
                else xs_,
                conts,
                key_chains,
                to_apply,
                prune_unapplied,
                map_nests=map_nests,
            ),
            out=out,
        )

    def repeat(
        self: ivy.Container,
        repeats: Union[int, Iterable[int]],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.repeat(x_, repeats=repeats, axis=axis)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def tile(
        self: ivy.Container,
        reps: Iterable[int],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.tile(x_, reps=reps) if ivy.is_array(x_) else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def constant_pad(
        self: ivy.Container,
        pad_width: Iterable[Tuple[int]],
        value: Number = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.constant_pad(x_, pad_width=pad_width, value=value)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def zero_pad(
        self: ivy.Container,
        pad_width: Iterable[Tuple[int]],
        value: Number = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.zero_pad(x_, pad_width=pad_width)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    def swapaxes(
        self: ivy.Container,
        axis0: int,
        axis1: int,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.handle_inplace(
            self.map(
                lambda x_, _: ivy.swapaxes(x_, axis0=axis0, axis1=axis1)
                if ivy.is_array(x_)
                else x_,
                key_chains,
                to_apply,
                prune_unapplied,
                map_sequences,
            ),
            out=out,
        )

    @staticmethod
    def static_clip(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        x_min: Optional[Union[Number, ivy.Array, ivy.NativeArray]] = None,
        x_max: Optional[Union[Number, ivy.Array, ivy.NativeArray]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.clip. This method simply wraps the
        function, and so the docstring for ivy.clip also applies to this method
        with minimal changes.

        Parameters
        ----------
        x
            Input array or container containing elements to clip.
        x_min
            Minimum value.
        x_max
            Maximum value.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of x, but where values < x_min are replaced
            with x_min, and those > x_max with x_max.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                              b=ivy.array([3., 4., 5.]))
        >>> y = ivy.Container.static_clip(x, 1., 5.)
        >>> print(y)
        {
            a: ivy.array([1., 1., 2.]),
            b: ivy.array([3., 4., 5.])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), \
                              b=ivy.array([3., 4., 5.]))
        >>> x_min = ivy.Container(a=1, b=-1)
        >>> x_max = ivy.Container(a=1, b=-1)
        >>> y = ivy.Container.static_roll(x, x_min, x_max)
        >>> print(y)
        {
            a: ivy.array([1., 1., 1.]),
            b: ivy.array([-1., -1., -1.])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "clip",
            x,
            x_min,
            x_max,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def clip(
        self: ivy.Container,
        x_min: Optional[Union[Number, ivy.Array, ivy.NativeArray]] = None,
        x_max: Optional[Union[Number, ivy.Array, ivy.NativeArray]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.clip. This method simply wraps the
        function, and so the docstring for ivy.clip also applies to this method
        with minimal changes.

        Parameters
        ----------
        self
            Input container containing elements to clip.
        x_min
            Minimum value.
        x_max
            Maximum value.
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
        out
            optional output container, for writing the result to. It must have a shape
            that the inputs broadcast to.

        Returns
        -------
        ret
            A container with the elements of x, but where values < x_min are replaced
            with x_min, and those > x_max with x_max.

        Examples
        --------
        >>> x = ivy.Container(a=ivy.array([0., 1., 2.]), b=ivy.array([3., 4., 5.]))
        >>> y = x.clip(1,2)
        >>> print(y)
        {
            a: ivy.array([1., 1., 2.]),
            b: ivy.array([2., 2., 2.])
        }
        """
        return self.static_clip(
            self,
            x_min,
            x_max,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

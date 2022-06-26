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
            out,
        )

    def expand_dims(
        self: ivy.Container,
        axis: Optional[int] = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
            out,
        )

    def permute_dims(
        self: ivy.Container,
        axes: Tuple[int, ...],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
            out,
        )

    def flip(
        self: ivy.Container,
        axis: Optional[Union[int, Tuple[int], List[int]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
            out,
        )

    def reshape(
        self: ivy.Container,
        shape: Tuple[int, ...],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
            out,
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
            out,
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
            out,
        )

    def repeat(
        self: ivy.Container,
        repeats: Union[int, Iterable[int]],
        axis: Optional[Union[int, Tuple[int, ...]]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
            out,
        )

    def tile(
        self: ivy.Container,
        reps: Iterable[int],
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
            out,
        )

    def constant_pad(
        self: ivy.Container,
        pad_width: Iterable[Tuple[int]],
        value: Number = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
            out,
        )

    def zero_pad(
        self: ivy.Container,
        pad_width: Iterable[Tuple[int]],
        value: Number = 0,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
            out,
        )

    def swapaxes(
        self: ivy.Container,
        axis0: int,
        axis1: int,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
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
            out,
        )

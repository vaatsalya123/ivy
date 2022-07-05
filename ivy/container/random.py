# global
from typing import Optional, Union, List, Dict, Tuple

# local
import ivy
from ivy.container.base import ContainerBase

# ToDo: implement all methods here as public instance methods


# noinspection PyMissingConstructor
class ContainerWithRandom(ContainerBase):
    @staticmethod
    def static_random_uniform(
        low: Union[float, ivy.Container] = 0.0,
        high: Union[float, ivy.Container] = 1.0,
        shape: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "random_uniform",
            low,
            high,
            shape,
            device=device,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def random_uniform(
        self: ivy.Container,
        low: Union[float, ivy.Container] = 0.0,
        high: Union[float, ivy.Container] = 1.0,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_random_uniform(
            low,
            high,
            self,
            device,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
            out=out,
        )

    # randint
    @staticmethod
    def static_randint(
        low: Union[int, ivy.Container] = 0.0,
        high: Union[int, ivy.Container] = 1.0,
        shape: Optional[Union[int, Tuple[int, ...], ivy.Container]] = None,
        device: Optional[Union[ivy.Device, ivy.NativeDevice, ivy.Container]] = None,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.randint. This method simply wraps the
        function, and so the docstring for ivy.randint also applies to this method
        with minimal changes.

        Examples
        --------
        With one :code:`ivy.Container` input:

        >>> x = ivy.Container.randint(low=ivy.Container(a=1, b=10), high=20, shape=2)
        >>> print(x)
        {
            a: ivy.array([10, 15]),
            b: ivy.array([16, 12])
        }

        >>> x = ivy.Container.randint(low=ivy.Container(a=1, b=4), high=15, shape=(3,2))
        >>> print(x)
        {
            a: ivy.array([[12, 3],
                         [5, 7],
                         [7, 2]]),
            b: ivy.array([[8, 10],
                         [9, 6],
                         [6, 7]])
        }

        >>> x = ivy.Container.randint(low=ivy.Container(a=5,b=20,c=40),\
                                      high=100,\
                                      shape=3,\
                                      device='gpu:1')
        >>> print(x)
        {
            a: ivy.array([90, 87, 62]),
            b: ivy.array([52, 95, 37]),
            c: ivy.array([95, 90, 42])
        }

        >>> x = ivy.Container(a=1,b=2)
        >>> y = ivy.Container.randint(low=ivy.Container(a=3,b=5,c=10,d=7),\
                                      high=14,\
                                      shape=5,\
                                      out=x)
        >>> print(x)
        {
            a: ivy.array([4, 10, 13, 3, 3]),
            b: ivy.array([12, 11, 11, 12, 5]),
            c: ivy.array([10, 13, 11, 13, 12]),
            d: ivy.array([12, 7, 8, 11, 8])
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container.randint(low=ivy.Container(a=1, b=10),\
                                      high=ivy.Container(a=5, b= 15, c=2),\
                                      shape=2)
        >>> print(x)
        {
            a: ivy.array([1, 2]),
            b: ivy.array([14, 10])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "randint",
            low,
            high,
            shape,
            device=device,
            key_chains=key_chains,
            out=out,
        )

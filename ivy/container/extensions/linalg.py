# global
from typing import Union, Optional, List, Dict

# local
from ivy.container.base import ContainerBase
import ivy


class ContainerWithLinalgExtensions(ContainerBase):
    @staticmethod
    def static_diagflat(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        offset: Optional[int] = 0,
        padding_value: Optional[float] = 0,
        align: Optional[str] = "RIGHT_LEFT",
        num_rows: Optional[int] = -1,
        num_cols: Optional[int] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "diagflat",
            x,
            offset=offset,
            padding_value=padding_value,
            align=align,
            num_rows=num_rows,
            num_cols=num_cols,
            out=out,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def diagflat(
        self: ivy.Container,
        /,
        *,
        offset: Optional[int] = 0,
        padding_value: Optional[float] = 0,
        align: Optional[str] = "RIGHT_LEFT",
        num_rows: Optional[int] = -1,
        num_cols: Optional[int] = -1,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.diagflat.
        This method simply wraps the function, and so the docstring for
        ivy.diagflat also applies to this method with minimal changes.

        Examples
        --------
        >>> x = ivy.Container(a=[1,2])
        >>> ivy.diagflat(x, k=1)
        {
            a: ivy.array([[0, 1, 0],
                          [0, 0, 2],
                          [0, 0, 0]])
        }
        """
        return self.static_diagflat(
            self,
            offset=offset,
            padding_value=padding_value,
            align=align,
            num_rows=num_rows,
            num_cols=num_cols,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    @staticmethod
    def static_kron(
        a: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        b: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container static method variant of ivy.kron. This method simply wraps
        the function, and so the docstring for ivy.kron also applies to this method
        with minimal changes.

        Parameters
        ----------
        a
            first container with input arrays.
        b
            second container with input arrays
        out
            optional output container, for writing the result to.

        Returns
        -------
        ret
            container including arrays corresponding to the Kronecker product of
            the arrays in the input containers, computed element-wise

        Examples
        --------
        >>> a = ivy.Container(x=ivy.array([1,2]), y=ivy.array(50))
        >>> b = ivy.Container(x=ivy.array([3,4]), y=ivy.array(9))
        >>> ivy.Container.static_kron(a, b)
        {
            a: ivy.array([3, 4, 6, 8])
            b: ivy.array([450])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "kron",
            a,
            b,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

    def kron(
        self: ivy.Container,
        b: ivy.Container,
        /,
        *,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.kron.
        This method simply wraps the function, and so the docstring for
        ivy.kron also applies to this method with minimal changes.

        Examples
        --------
        >>> a = ivy.Container(x=ivy.array([1,2]), y=ivy.array([50]))
        >>> b = ivy.Container(x=ivy.array([3,4]), y=ivy.array(9))
        >>> a.kron(b)
        {
            a: ivy.array([3, 4, 6, 8])
            b: ivy.array([450])
        }
        """
        return self.static_kron(
            self,
            b,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
            out=out,
        )

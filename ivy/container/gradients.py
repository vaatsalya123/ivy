from typing import Optional, Union, List, Dict

# local
import ivy
from ivy.container.base import ContainerBase


# noinspection PyMissingConstructor
class ContainerWithGradients(ContainerBase):
    @staticmethod
    def static_is_variable(
        x: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        exclusive: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.is_variable.
        This method simply wraps the function, and so the docstring for
        ivy.is_variable also applies to this method with minimal changes

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([2, -1, 0]), b=ivy.array([0., -0.4, 8]))
        >>> is_var = ivy.Container.static_is_variable(x)
        >>> print(is_var)
        {
            a: false,
            b: false
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([2, -1, 0]), b=ivy.array([0., -0.4, 8]))
        >>> exclusive = ivy.Container(a=False, b=True)
        >>> is_var = ivy.Container.static_is_variable(x, exclusive=exclusive)
        >>> print(is_var)
        {
            a: false,
            b: false
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "is_variable",
            x,
            exclusive,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def is_variable(
        self: ivy.Container,
        exclusive: Union[bool, ivy.Container] = False,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.is_variable.
        This method simply wraps the function, and so the docstring for
        ivy.is_variable also applies to this method with minimal changes

        Examples
        --------
        With :code:`ivy.Container` input:

        >>> x = ivy.Container(a=ivy.array([2, -1, 0]), b=ivy.array([0., -0.4, 8]))
        >>> is_var = x.is_variable(exclusive=True)
        >>> print(is_var)
        {
            a: false,
            b: false
        }

        With multiple :code:`ivy.Container` inputs:

        >>> x = ivy.Container(a=ivy.array([2, -1, 0]), b=ivy.array([0., -0.4, 8]))
        >>> exclusive = ivy.Container(a=True, b=True)
        >>> is_var = x.is_variable(exclusive=exclusive)
        >>> print(is_var)
        {
            a: false,
            b: false
        }
        """
        return self.static_is_variable(
            self, exclusive, key_chains, to_apply, prune_unapplied, map_sequences
        )

    @staticmethod
    def static_adam_step(
        dcdw: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        mw: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        vw: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        step: Union[int, float],
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.adam_step.
        This method simply wraps the function, and so the docstring for ivy.
        adam_step also applies to this method with minimal changes

        Examples
        --------
        with :code: `ivy.container` inputs:

        >>> dcdw = ivy.Container(a=ivy.array([0., 1., 2.]),\
                         b=ivy.array([3., 4., 5.]))
        >>> mw = ivy.Container(a=ivy.array([0., 0., 0.]),\
                               b=ivy.array([0., 0., 0.]))
        >>> vw = ivy.Container(a=ivy.array([0.,]),\
                               b=ivy.array([0.,]))
        >>> step = ivy.array([3.4])
        >>> beta1 = 0.87
        >>> beta2 = 0.976
        >>> epsilon = 1e-5
        >>> adam_step_delta = ivy.Container.static_adam_step(dcdw,\
                                mw, vw, step, beta1, beta2, epsilon)
        >>> print(adam_step_delta)
        {
            a: (list[3], <class ivy.array.Array> shape=[3]),
            b: (list[3], <class ivy.array.Array> shape=[3])
        }
        """
        return ContainerBase.multi_map_in_static_method(
            "adam_update",
            dcdw,
            mw,
            vw,
            step,
            beta1,
            beta2,
            epsilon,
        )

    def adam_step(
        self: ivy.Container,
        mw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        vw: Union[ivy.Array, ivy.NativeArray, ivy.Container],
        step: Union[int, float],
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
    ) -> ivy.Container:
        """
        ivy.Container instance method variant of ivy.adam_step.
        This method simply wraps the function, and so the docstring for ivy.
        adam_step also applies to this method with minimal changes

        Examples
        --------
        with :code: `ivy.container` inputs:

        >>> dcdw = ivy.Container(a=ivy.array([0., 1., 2.]),\
                         b=ivy.array([3., 4., 5.]))
        >>> mw = ivy.Container(a=ivy.array([0., 0., 0.]),\
                               b=ivy.array([0., 0., 0.]))
        >>> vw = ivy.Container(a=ivy.array([0.,]),\
                               b=ivy.array([0.,]))
        >>> step = ivy.array(1)
        >>> adam_step_delta = dcdw.adam_step(mw, vw, step, beta1, beta2, epsilon)
        {
            a: (list[3], <class ivy.array.Array> shape=[3]),
            b: (list[3], <class ivy.array.Array> shape=[3])
        }
        """
        return self.static_adam_step(self, mw, vw, step, beta1, beta2, epsilon)

    @staticmethod
    def static_optimizer_update(
        w,
        effective_grad,
        lr,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "optimizer_update",
            w,
            effective_grad,
            lr,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def optimizer_update(
        self: ivy.Container,
        effective_grad,
        lr,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return self.static_optimizer_update(
            self,
            effective_grad,
            lr,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_gradient_descent_update(
        w,
        dcdw,
        lr,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:

        return ContainerBase.multi_map_in_static_method(
            "gradient_descent_update",
            w,
            dcdw,
            lr,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def gradient_descent_update(
        self,
        dcdw,
        lr,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ):
        return self.static_gradient_descent_update(
            self,
            dcdw,
            lr,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_lars_update(
        w,
        dcdw,
        lr,
        decay_lambda=0,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "lars_update",
            w,
            dcdw,
            lr,
            decay_lambda=decay_lambda,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def lars_update(
        self,
        dcdw,
        lr,
        decay_lambda=0,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ):
        return self.static_lars_update(
            self,
            dcdw,
            lr,
            decay_lambda,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_adam_update(
        w,
        dcdw,
        lr,
        mw_tm1,
        vw_tm1,
        step,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "adam_update",
            w,
            dcdw,
            lr,
            mw_tm1=mw_tm1,
            vw_tm1=vw_tm1,
            step=step,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def adam_update(
        self,
        dcdw,
        lr,
        mw_tm1,
        vw_tm1,
        step,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ):
        return self.static_adam_update(
            self,
            dcdw,
            lr,
            mw_tm1,
            vw_tm1,
            step,
            beta1,
            beta2,
            epsilon,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

    @staticmethod
    def static_lamb_update(
        w,
        dcdw,
        lr,
        mw_tm1,
        vw_tm1,
        step,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
        max_trust_ratio=10,
        decay_lambda=0,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ) -> ivy.Container:
        return ContainerBase.multi_map_in_static_method(
            "lamb_update",
            w,
            dcdw,
            lr,
            mw_tm1=mw_tm1,
            vw_tm1=vw_tm1,
            step=step,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            max_trust_ratio=max_trust_ratio,
            decay_lambda=0,
            inplace=inplace,
            stop_gradients=stop_gradients,
            key_chains=key_chains,
            to_apply=to_apply,
            prune_unapplied=prune_unapplied,
            map_sequences=map_sequences,
        )

    def lamb_update(
        self,
        dcdw,
        lr,
        mw_tm1,
        vw_tm1,
        step,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-7,
        max_trust_ratio=10,
        decay_lambda=0,
        inplace=None,
        stop_gradients=True,
        key_chains: Optional[Union[List[str], Dict[str, str]]] = None,
        to_apply: bool = True,
        prune_unapplied: bool = False,
        map_sequences: bool = False,
    ):
        return self.static_lamb_update(
            self,
            dcdw,
            lr,
            mw_tm1,
            vw_tm1,
            step,
            beta1,
            beta2,
            epsilon,
            max_trust_ratio,
            decay_lambda,
            inplace,
            stop_gradients,
            key_chains,
            to_apply,
            prune_unapplied,
            map_sequences,
        )

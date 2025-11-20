from typing import Dict, Optional, Tuple, Union, Sequence
import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass
import equinox as eqx


class BaseModel(eqx.Module):

    state: jnp.ndarray
    dt: float = 0.1
    t: float = 0.0

    def reset(self, state: Optional[jnp.ndarray] = None):
        if state is not None:
            self.state = state
        self.t = 0.0

    def _dynamics(self, state: jnp.ndarray, t: float) -> jnp.ndarray:
        raise NotImplementedError

    def _step(self, state: jnp.ndarray, t: float) -> Tuple[jnp.ndarray, float]:
        deriv = self._dynamics(state, t)
        new_state = state + self.dt * deriv
        return new_state, t + self.dt

    def simulate(self, duration: float, steps: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        n_steps = int(jnp.ceil(duration / self.dt))

        def body(carry, _):
            state, t = carry
            new_state, new_t = self._step(state, t)
            return (new_state, new_t), (new_t, new_state)

        (_, _), (times, states) = lax.scan(body, (self.state, self.t), None, length=n_steps)
        times = jnp.concatenate([jnp.array([self.t]), times])
        states = jnp.vstack([self.state, states])
        return times, states

    def derivative_model(self, x: float, with_respect_to: Union[str, Sequence[str]]) -> "BaseModel":
        """
        Computes derivatives w.r.t. a flexible subset of parameters using filtering.

        Args:
            x: The input value.
            with_respect_to: A string or sequence of strings (e.g., 'a', or ['a', 'c']).

        Returns:
            A new BaseModel instance where attributes corresponding to the
            requested parameters contain gradients, and all others are None.
        """
        # 1. Standardize the input
        if isinstance(with_respect_to, str):
            params_to_diff = (with_respect_to,)
        else:
            params_to_diff = tuple(with_respect_to)

        # 2. Define a filter function that checks if a leaf of the PyTree
        #    is one of the attributes the user wants to differentiate.
        #    We do this by checking for object identity.
        def is_differentiable(leaf):
            return any(leaf is getattr(self, name) for name in params_to_diff)

        # 3. Partition the model based on the user's request.
        diff_model, static_model = eqx.partition(self, is_differentiable)

        # 4. Define the loss function for the partitioned model.
        def loss_for_grad(differentiable_part, static_part, x_in):
            model = eqx.combine(differentiable_part, static_part)
            return model.simulate(x_in)[1].sum()

        # 5. Differentiate only with respect to the `differentiable_part`.
        grad_fn = jax.grad(loss_for_grad, argnums=0)

        # 6. Execute and return the resulting gradient model.
        grad_of_diff_part = grad_fn(diff_model, static_model, x)
        return grad_of_diff_part

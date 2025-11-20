from typing import Dict, Optional, Tuple, Union, Sequence
import jax
import jax.numpy as jnp
from jax import lax
import diffrax
from dataclasses import dataclass
import equinox as eqx
from .model_utils import compute_delay_matrix


class BaseModel(eqx.Module):

    state: jnp.ndarray
    fiber_count_matrix: jnp.ndarray
    connectivity_matrix: jnp.ndarray
    fiber_length_matrix: jnp.ndarray
    signal_propagation_speed: float
    delay_matrix: jnp.ndarray
    number_of_regions: int
    K_gl: float = 1.0
    dt: float
    t: float = 0.0
    key: jnp.ndarray

    def __init__(
        self,
        state,
        dt=0.1,
        fiber_count_matrix=jnp.ones((1, 1)),
        fiber_length_matrix=jnp.zeros((1, 1)),
        signal_propagation_speed: float = 20.0,
        seed=42,
    ):
        self.state = state
        self.dt = dt
        self.fiber_count_matrix = fiber_count_matrix
        self.connectivity_matrix = jnp.fill_diagonal(self.fiber_count_matrix, 0.0, inplace=False)
        self.fiber_length_matrix = fiber_length_matrix
        self.signal_propagation_speed = signal_propagation_speed
        for matrix in (self.fiber_length_matrix, self.fiber_count_matrix):
            assert len(matrix.shape) == 2
            assert matrix.shape[0] == matrix.shape[1]
        self.number_of_regions = self.fiber_count_matrix.shape[0]
        self.delay_matrix = compute_delay_matrix(self.fiber_length_matrix, self.signal_propagation_speed)
        self.key = jax.random.PRNGKey(seed) if seed is not None else jax.random.PRNGKey(0)

    def reset(self, state: Optional[jnp.ndarray] = None):
        if state is not None:
            self.state = state
        self.t = 0.0

    def dynamics(self, t, y, args, *, history):
        return NotImplementedError

    def history_fn(self, t):
        return NotImplementedError

    def simulate(self, duration=50.0, steps=1000):
        t0, t1 = 0.0, duration
        ts = jnp.linspace(t0, t1, steps)

        term = diffrax.ODETerm(self.dynamics)
        solver = diffrax.Bosh3()

        delays = diffrax.Delays(
            delays=[lambda t, y, args: d for d in self.delay_matrix.flatten()],
            initial_discontinuities=jnp.array([0.0]),
        )

        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t0,
            t1=t1,
            dt0=0.01,
            y0=self.history_fn,
            args=None,
            saveat=diffrax.SaveAt(ts=ts, dense=True),
            stepsize_controller=diffrax.PIDController(
                rtol=1e-3,
                atol=1e-6,
            ),
            delays=delays,
            max_steps=16**5,
        )
        return sol.ts, sol.ys

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

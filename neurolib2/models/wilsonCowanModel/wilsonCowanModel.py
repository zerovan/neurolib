import jax
import jax.numpy as jnp
from typing import Dict, Optional
import diffrax
from dataclasses import dataclass
from ..base.base import BaseModel
import matplotlib.pyplot as plt


class WilsonCowan(BaseModel):

    tau_EI: jax.Array
    tau_IE: jax.Array
    tau_e: jax.Array
    tau_i: jax.Array
    w_ee: jax.Array
    w_ei: jax.Array
    w_ie: jax.Array
    w_ii: jax.Array
    P_e: jax.Array
    P_i: jax.Array
    a_e: jax.Array
    a_i: jax.Array
    theta_e: jax.Array
    theta_i: jax.Array

    def __init__(
        self,
        state: jnp.ndarray,
        dt: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(state=state, dt=dt, *args, **kwargs)
        self.tau_EI = jnp.array(1.0)
        self.tau_IE = jnp.array(2.0)
        self.tau_e = jnp.array(0.1)
        self.tau_i = jnp.array(0.5)
        self.w_ee = jnp.array(16.0)
        self.w_ei = jnp.array(12.0)
        self.w_ie = jnp.array(15.0)
        self.w_ii = jnp.array(3.0)
        self.P_e = jnp.array(1.0)
        self.P_i = jnp.array(0.0)
        self.a_e = jnp.array(1.0)
        self.a_i = jnp.array(1.0)
        self.theta_e = jnp.array(2.0)
        self.theta_i = jnp.array(2.0)

    def _logistic(self, x: jnp.ndarray, a: float, theta: float) -> jnp.ndarray:
        return 1.0 / (1.0 + jnp.exp(-a * (x - theta)))

    def _S_e(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._logistic(x, self.a_e, self.theta_e)

    def _S_i(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._logistic(x, self.a_i, self.theta_i)

    def dynamics(self, t, y, args, *, history):
        """
        history shape: (num_delays, len(y), number_of_regions)
        """
        exc, inh = y

        # unflatten the delays
        history = jnp.array(history).reshape(
            self.number_of_regions, self.number_of_regions, len(y), self.number_of_regions
        )  # TODO: confirm that reshaping works correctly
        history = jnp.nan_to_num(history, nan=0.0)  # TODO: why did we have nan's?
        # jax.debug.print("{}", history)

        # TODO: get rid of loops
        exc_interareal_input = []
        for to_region in range(self.number_of_regions):
            row_sum = 0.0
            # ? row_sum = self.connectivity_matrix[to_region] * jnp.diagonal(history[to_region, :, 0])
            for from_region in range(self.number_of_regions):
                row_sum += (
                    self.connectivity_matrix[to_region, from_region] * history[to_region, from_region][0][from_region]
                )
            exc_interareal_input.append(row_sum)
        exc_interareal_input = jnp.array(exc_interareal_input)
        # jax.debug.print("{}", exc_interareal_input)

        exc_rhs = (
            1
            / self.tau_e
            * (
                -exc
                + (1 - exc)
                * self._S_e(
                    self.w_ee * exc  # input from within the excitatory population
                    - self.w_ie * inh  # input from the inhibitory population
                    + exc_interareal_input  # input from other nodes
                    # TODO + exc_ext_baseline  # baseline external input (static)
                    # TODO + exc_ext[:, i]  # time-dependent external input
                )
                # TODO + exc_ou  # ou noise
            )
        )
        inh_rhs = (
            1
            / self.tau_i
            * (
                -inh
                + (1 - inh)
                * self._S_i(
                    self.w_ei * exc  # input from the excitatory population
                    - self.w_ii * inh  # input from within the inhibitory population
                    # TODO + inh_ext_baseline  # baseline external input (static)
                    # TODO + inh_ext[:, i]  # time-dependent external input
                )
                # TODO + inh_ou  # ou noise
            )
        )
        return exc_rhs, inh_rhs

    def history_fn(self, t):
        return jnp.zeros(self.number_of_regions, dtype=float), jnp.zeros(self.number_of_regions, dtype=float)

    def plot(self, times: jnp.ndarray, states: jnp.ndarray, show: bool = True):
        # plot activity of first node
        E = states[0, :, 0]
        I = states[1, :, 0]
        plt.figure(figsize=(9, 4))
        plt.plot(times, E, label="E (exc)", linewidth=1.5)
        plt.plot(times, I, label="I (inh)", linewidth=1.5)
        plt.legend()
        plt.xlabel("time (s)")
        plt.ylabel("activity")
        plt.legend()
        plt.title("Wilson-Cowan dynamics")
        plt.tight_layout()
        if show:
            plt.show()

    @staticmethod
    def create_default(dt: float = 0.1, initial_state: Optional[jnp.ndarray] = None):
        if initial_state is None:
            initial_state = jnp.array([0.1, 0.1])

        fiber_length_matrix = jnp.array(
            [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
            ]
        ).astype(float)
        fiber_count_matrix = jnp.array(
            [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
            ]
        ).astype(float)
        return WilsonCowan(
            state=initial_state, dt=dt, fiber_length_matrix=fiber_length_matrix, fiber_count_matrix=fiber_count_matrix
        )

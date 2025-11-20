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
        # print(len(history[0][0]))
        # exc_delayed = history[0].reshape(self.number_of_regions, self.number_of_regions)
        # exc_input = jnp.sum(self.K_gl * self.connectivity_matrix * exc_history[range_N, -Dmat_ndt - 1], axis=1)
        region_selection = []
        for i in range(self.number_of_regions):
            region_selection += [i]
        region_selection = jnp.array(region_selection)
        exc_delayed = jnp.array(history)[:, 0, region_selection]
        print(exc_delayed.shape)
        return jnp.zeros(self.number_of_regions), jnp.zeros(self.number_of_regions)
        """
        # E_delayed, I_delayed = history[0]
        # input_e = self.w_ee * E_delayed - self.w_ei * I + self.P_e
        # input_i = self.w_ie * E - self.w_ii * I_delayed + self.P_i

        delayed_EI, delayed_IE = history

        E_tau_EI = delayed_EI[0]  # E(t - tau_EI)
        I_tau_IE = delayed_IE[1]  # I(t - tau_IE)

        input_e = self.w_ee * E - self.w_ei * I_tau_IE + self.P_e
        input_i = self.w_ie * E_tau_EI - self.w_ii * I + self.P_i

        S_e = self._S_e(input_e)
        S_i = self._S_i(input_i)

        dE = (-E + S_e) / self.tau_e
        dI = (-I + S_i) / self.tau_i
        return jnp.array([dE, dI])
        """

    def history_fn(self, t):
        return jnp.zeros(self.number_of_regions), jnp.zeros(self.number_of_regions)

    def plot(self, times: jnp.ndarray, states: jnp.ndarray, show: bool = True):
        E = states[:, 0]
        I = states[:, 1]
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
        )
        fiber_count_matrix = jnp.array(
            [
                [1, 2, 3],
                [1, 2, 3],
                [1, 2, 3],
            ]
        )
        return WilsonCowan(
            state=initial_state, dt=dt, fiber_length_matrix=fiber_length_matrix, fiber_count_matrix=fiber_count_matrix
        )

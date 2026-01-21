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
    
    sigma_e: jax.Array
    sigma_i: jax.Array
    
    
    exc_ext_baseline: jax.Array
    inh_ext_baseline: jax.Array

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
        
        self.sigma_e = jnp.array(0.5)
        self.sigma_i = jnp.array(0.5)
        
        self.exc_ext_baseline = jnp.array(0.0)
        self.inh_ext_baseline = jnp.array(0.0)

    def _logistic(self, x: jnp.ndarray, a: float, theta: float) -> jnp.ndarray:
        return 1.0 / (1.0 + jnp.exp(-a * (x - theta)))

    def _S_e(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._logistic(x, self.a_e, self.theta_e)

    def _S_i(self, x: jnp.ndarray) -> jnp.ndarray:
        return self._logistic(x, self.a_i, self.theta_i)

    def dynamics(self, t, y, args, *, history):
        """
        history shape: (num_delays, len(y), number_of_regions)(9,2,3)
        """
        exc, inh = y
        history = jnp.array(history)
        history = jnp.nan_to_num(history, nan=0.0)  # TODO: why did we have nan's?
        # jax.debug.print("{}", history)

        delayed_exc = history[
            self.delay_index_matrix,  # (N, N)
            0,                        # excitatory population
            jnp.arange(self.number_of_regions)[None, :]  # from_region
        ]
        
        exc_interareal_input = jnp.sum(
            self.connectivity_matrix * delayed_exc,
            axis=1
        )

        # one noise draw per step (shared or separate — see note below)
        key = jax.random.fold_in(self.key, jnp.floor(t / self.dt).astype(jnp.int32))
        key_e, key_i = jax.random.split(key)
        noise_e = jax.random.normal(key_e, (self.number_of_regions,))
        noise_i = jax.random.normal(key_i, (self.number_of_regions,))

        exc_rhs_det = (
            1
            / self.tau_e
            * (
                -exc
                + (1 - exc)
                * self._S_e(
                    self.w_ee * exc  # input from within the excitatory population
                    - self.w_ie * inh  # input from the inhibitory population
                    + exc_interareal_input  # input from other nodes
                    + self.exc_ext_baseline  # baseline external input (static)
                    # TODO + exc_ext[:, i]  # time-dependent external input
                )
            )
        )
        inh_rhs_det = (
            1
            / self.tau_i
            * (
                -inh
                + (1 - inh)
                * self._S_i(
                    self.w_ei * exc  # input from the excitatory population
                    - self.w_ii * inh  # input from within the inhibitory population
                    + self.inh_ext_baseline  # baseline external input (static)
                    # TODO + inh_ext[:, i]  # time-dependent external input
                )
            )
        )
        
        exc_rhs = exc_rhs_det + self.sigma_e * jnp.sqrt(self.dt) * noise_e
        inh_rhs = inh_rhs_det + self.sigma_i * jnp.sqrt(self.dt) * noise_i
        return exc_rhs, inh_rhs 

    def history_fn(self, t):
        zeros = jnp.zeros(self.number_of_regions, dtype=float)
        return zeros, zeros

    def plot(self, times: jnp.ndarray, states: jnp.ndarray, show: bool = True):
        num_regions = self.number_of_regions
        plt.figure(figsize=(10, 5))

        for r in range(num_regions):
            plt.plot(times, states[0, :, r], label=f"E node {r}")
            plt.plot(times, states[1, :, r], label=f"I node {r}", linestyle="--")

        plt.xlabel("time (s)")
        plt.ylabel("activity")
        plt.title("Wilson-Cowan: all node activities")
        plt.legend()
        plt.tight_layout()

        if show:
            plt.show()


    @staticmethod
    def create_default(dt: float = 0.1, fiber_length_matrix= None, fiber_count_matrix= None, initial_state: Optional[jnp.ndarray] = None):
        if initial_state is None:
            # (E, I)
            initial_state = jnp.array([0.1, 0.1])

        if fiber_length_matrix is None:
            fiber_length_matrix = jnp.array(
                [
                    [1, 2, 3],
                    [1, 2, 3],
                    [1, 2, 3],
                ]
            ).astype(float)
        if fiber_count_matrix is None:
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

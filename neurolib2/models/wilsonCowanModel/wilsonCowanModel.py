import jax
import jax.numpy as jnp
from typing import Dict, Optional
import diffrax
from dataclasses import dataclass
from ..base.base import BaseModel
from ..base.lineax_utils import OULinearOperator
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

    sigma_ou: jax.Array
    tau_ou: jax.Array
    mean_exc_ou: jax.Array
    mean_inh_ou: jax.Array

    exc_ext_baseline: jax.Array
    inh_ext_baseline: jax.Array

    populations_per_region = 2

    def __init__(
        self,
        state: jnp.ndarray,
        dt: float = 0.1,
        tau_EI: float = 1.0,
        tau_IE: float = 2.0,
        tau_e: float = 0.1,
        tau_i: float = 0.5,
        w_ee: float = 16.0,
        w_ei: float = 12.0,
        w_ie: float = 15.0,
        w_ii: float = 3.0,
        P_e: float = 1.0,
        P_i: float = 0.0,
        a_e: float = 1.0,
        a_i: float = 1.0,
        theta_e: float = 2.0,
        theta_i: float = 2.0,
        sigma_ou: float = 0.5,
        tau_ou: float = 5.0,
        mean_exc_ou: float = 0.0,
        mean_inh_ou: float = 0.0,
        exc_ext_baseline: float = 0.0,
        inh_ext_baseline: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__(state=state, dt=dt, *args, **kwargs)
        self.tau_EI = jnp.array(tau_EI)
        self.tau_IE = jnp.array(tau_IE)
        self.tau_e = jnp.array(tau_e)
        self.tau_i = jnp.array(tau_i)
        self.w_ee = jnp.array(w_ee)
        self.w_ei = jnp.array(w_ei)
        self.w_ie = jnp.array(w_ie)
        self.w_ii = jnp.array(w_ii)
        self.P_e = jnp.array(P_e)
        self.P_i = jnp.array(P_i)
        self.a_e = jnp.array(a_e)
        self.a_i = jnp.array(a_i)
        self.theta_e = jnp.array(theta_e)
        self.theta_i = jnp.array(theta_i)

        self.sigma_ou = jnp.array(sigma_ou)
        self.tau_ou = jnp.array(tau_ou)
        self.mean_exc_ou = jnp.array(mean_exc_ou)
        self.mean_inh_ou = jnp.array(mean_inh_ou)

        self.exc_ext_baseline = jnp.array(exc_ext_baseline)
        self.inh_ext_baseline = jnp.array(inh_ext_baseline)

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
            0,  # excitatory population
            jnp.arange(self.number_of_regions)[None, :],  # from_region
        ]

        exc_interareal_input = jnp.sum(self.connectivity_matrix * delayed_exc, axis=1)

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

        return jnp.array((exc_rhs_det, inh_rhs_det))

    def get_term(self, ts):
        # dynamics: function, shape (k, n)
        # noise: MultiTerm, shape (k, n)
        # returns: MultiTerm, shape (2k, n)
        ou_means = jnp.array([[self.mean_exc_ou], [self.mean_inh_ou]])

        def drift(t, y, args, *, history=None):
            return -self.tau_ou * (y - ou_means)

        def stacked_drift(t, y, args, *, history=None):
            dynamics_state = y[: self.populations_per_region]
            ou_state = y[self.populations_per_region :]
            return jnp.vstack((self.dynamics(t, dynamics_state, args, history=history), drift(t, ou_state, args)))

        def diffusion(t, y, args, *, history=None):
            return OULinearOperator(
                self.sigma_ou,
                self.populations_per_region,
                self.number_of_regions,
            )

        brownian_motion = diffrax.VirtualBrownianTree(
            ts[0],
            ts[-1] + self.dt,
            tol=1e-3,
            shape=(
                self.populations_per_region,
                self.number_of_regions,
            ),
            key=self.key,
        )

        return diffrax.MultiTerm(diffrax.ODETerm(stacked_drift), diffrax.ControlTerm(diffusion, brownian_motion))

    def history_fn(self, t):
        return jnp.zeros((2 * self.populations_per_region, self.number_of_regions), dtype=float)

    def plot(self, times: jnp.ndarray, states: jnp.ndarray, show: bool = True):
        """
        Plot E/I activity. Accepts states in either:
        - time-major: (T, components, N)
        - component-major: (components, N, T)
        - single timepoint: (components, N)
        Where components >= 2 (E, I, ...).
        """
        # normalize to (T, components, N)
        states = jnp.asarray(states)
        if states.ndim == 3:
            # either (T, comp, N) or (comp, N, T)
            if states.shape[0] == len(times):
                states_time_major = states
            elif states.shape[2] == len(times):
                states_time_major = jnp.transpose(states, (2, 0, 1))
            else:
                raise ValueError(f"Cannot infer time axis from states with shape {states.shape}")
        elif states.ndim == 2:
            # single timepoint -> promote to length-1 time axis
            states_time_major = states[None, ...]
        else:
            raise ValueError(f"Unsupported states shape: {states.shape}")

        num_regions = states_time_major.shape[2]  # (T, components, N)
        plt.figure(figsize=(10, 5))

        for r in range(num_regions):
            plt.plot(times, states_time_major[:, 0, r], label=f"E node {r}")
            plt.plot(times, states_time_major[:, 1, r], label=f"I node {r}", linestyle="--")
            # plt.plot(times, states_time_major[:, 0, r] +  states_time_major[:, 2, r], label=f"E node {r}")
            # plt.plot(times, states_time_major[:, 1, r] +  states_time_major[:, 3, r], label=f"I node {r}", linestyle="--")

        plt.xlabel("time (s)")
        plt.ylabel("activity")
        plt.title("Wilson-Cowan: all node activities")
        plt.legend()
        plt.tight_layout()

        if show:
            plt.show()


    @staticmethod
    def create_default(
        dt: float = 0.1, fiber_length_matrix=None, fiber_count_matrix=None, initial_state: Optional[jnp.ndarray] = None
    ):
        if initial_state is None:
            # (E, I)
            initial_state = jnp.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

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

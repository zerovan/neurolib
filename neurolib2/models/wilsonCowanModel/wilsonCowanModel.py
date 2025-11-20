import jax
import jax.numpy as jnp
from typing import Dict, Optional
import diffrax
from dataclasses import dataclass
from ..base.baseModel import BaseModel
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
    ):
        super().__init__(state=state, dt=dt)
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



    def wc_vector_field_with_delay(self, t, y, args, *, history):
        E, I = y
        # E_delayed, I_delayed = history[0]
        # input_e = self.w_ee * E_delayed - self.w_ei * I + self.P_e
        # input_i = self.w_ie * E - self.w_ii * I_delayed + self.P_i


        delayed_EI, delayed_IE = history

        E_tau_EI = delayed_EI[0]   # E(t - tau_EI)
        I_tau_IE = delayed_IE[1]   # I(t - tau_IE)

        input_e = self.w_ee * E - self.w_ei * I_tau_IE + self.P_e
        input_i = self.w_ie * E_tau_EI - self.w_ii * I + self.P_i
        
        S_e = self._S_e(input_e)
        S_i = self._S_i(input_i)

        dE = (-E + S_e) / self.tau_e
        dI = (-I + S_i) / self.tau_i
        return jnp.array([dE, dI])

    def history_fn(self, t):
        return jnp.array([0.1, 0.1])

    
    def simulate(self, duration=50.0, steps=1000):
        t0, t1 = 0.0, duration
        ts = jnp.linspace(t0, t1, steps)

        term = diffrax.ODETerm(self.wc_vector_field_with_delay)
        solver = diffrax.Bosh3()


        delays = diffrax.Delays(
            delays=[
                lambda t, y, args: self.tau_EI,   # first delay for E→I
                lambda t, y, args: self.tau_IE,   # second delay for I→E
            ],
            initial_discontinuities=jnp.array([0.0])
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
            max_steps=5_000_000
        )
        return sol.ts, sol.ys

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

        params = dict()
        return WilsonCowan(state=initial_state, dt=dt)

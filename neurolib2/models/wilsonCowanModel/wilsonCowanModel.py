import jax
import jax.numpy as jnp
from typing import Dict, Optional
from dataclasses import dataclass
from ..base.baseModel import BaseModel
import matplotlib.pyplot as plt

@dataclass
class WilsonCowan(BaseModel):

    def _logistic(self, x: jnp.ndarray, a: float, theta: float) -> jnp.ndarray:
        return 1.0 / (1.0 + jnp.exp(-a * (x - theta)))

    def _S_e(self, x: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
        return self._logistic(x, params["a_e"], params["theta_e"])

    def _S_i(self, x: jnp.ndarray, params: Dict[str, float]) -> jnp.ndarray:
        return self._logistic(x, params["a_i"], params["theta_i"])

    def _dynamics(self, state: jnp.ndarray, t: float, params: Dict[str, float]) -> jnp.ndarray:
        E, I = state
        input_e = params["w_ee"] * E - params["w_ei"] * I + params["P_e"]
        input_i = params["w_ie"] * E - params["w_ii"] * I + params["P_i"]

        S_e = self._S_e(input_e, params)
        S_i = self._S_i(input_i, params)

        dE = (-E + S_e) / params["tau_e"]
        dI = (-I + S_i) / params["tau_i"]
        return jnp.array([dE, dI])
    
    def plot(self, times: jnp.ndarray, states: jnp.ndarray, show: bool = True):
        E = states[:, 0]
        I = states[:, 1]
        plt.figure(figsize=(9, 4))
        plt.plot(times, E, label="E (exc)", linewidth=1.5)
        plt.plot(times, I, label="I (inh)", linewidth=1.5)
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

        params = dict(
            tau_e=1.0,
            tau_i=40.0,
            w_ee=16.0,
            w_ei=12.0,
            w_ie=15.0,
            w_ii=3.0,
            P_e=1.0,
            P_i=0.0,
            a_e=1.0,
            a_i=1.0,
            theta_e=2.0,
            theta_i=2.0,
        )
        return WilsonCowan(state=initial_state, params=params, dt=dt)

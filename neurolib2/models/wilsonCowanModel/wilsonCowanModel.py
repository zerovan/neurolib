from ..base.baseModel import BaseModel
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

class WilsonCowan(BaseModel):
    """
    Wilson-Cowan two-population model.

    Equations (standard form, Euler integrates d/dt form):
        tau_e * dE/dt = -E + S_e( w_ee * E - w_ei * I + P_e )
        tau_i * dI/dt = -I + S_i( w_ie * E - w_ii * I + P_i )

    where S(x) is a sigmoid (default logistic).

    Parameters (defaults provided, you can override via params dict or set_params):
        tau_e, tau_i: time constants
        w_ee, w_ei, w_ie, w_ii: connection weights (positive numbers)
        P_e, P_i: external inputs (can be time-varying if you modify before each step)
        a_e, a_i: sigmoid gains
        theta_e, theta_i: sigmoid thresholds
        sigmoid: callable(x, params) -> output (if provided, overrides gain/threshold)
    """

    def __init__(self,
                 tau_e: float = 10.0,
                 tau_i: float = 20.0,
                 w_ee: float = 12.0,
                 w_ei: float = 10.0,
                 w_ie: float = 10.0,
                 w_ii: float = 0.0,
                 P_e: float = 0.0,
                 P_i: float = 0.0,
                 a_e: float = 1.0,
                 a_i: float = 1.0,
                 theta_e: float = 0.0,
                 theta_i: float = 0.0,
                 dt: float = 0.1,
                 initial_state: Optional[np.ndarray] = None):
        # state vector: [E, I]
        if initial_state is None:
            initial_state = np.array([0.1, 0.1], dtype=float)
        params = dict(
            tau_e=float(tau_e),
            tau_i=float(tau_i),
            w_ee=float(w_ee),
            w_ei=float(w_ei),
            w_ie=float(w_ie),
            w_ii=float(w_ii),
            P_e=float(P_e),
            P_i=float(P_i),
            a_e=float(a_e),
            a_i=float(a_i),
            theta_e=float(theta_e),
            theta_i=float(theta_i),
            sigmoid=None,  # if set to callable, used instead of logistic
        )
        super().__init__(initial_state=initial_state, params=params, dt=dt)

    @staticmethod
    def _logistic(x: np.ndarray, a: float = 1.0, theta: float = 0.0) -> np.ndarray:
        # logistic sigmoid: 1 / (1 + exp(-a*(x - theta)))
        return 1.0 / (1.0 + np.exp(-a * (x - theta)))

    def _S_e(self, x: np.ndarray) -> np.ndarray:
        sig = self.params.get("sigmoid")
        if callable(sig):
            return sig(x, **self.params)
        return self._logistic(x, a=self.params["a_e"], theta=self.params["theta_e"])

    def _S_i(self, x: np.ndarray) -> np.ndarray:
        sig = self.params.get("sigmoid")
        if callable(sig):
            return sig(x, **self.params)
        return self._logistic(x, a=self.params["a_i"], theta=self.params["theta_i"])

    def _dynamics(self, state: np.ndarray, t: float, params: Dict) -> np.ndarray:
        E, I = state
        tau_e = params["tau_e"]
        tau_i = params["tau_i"]
        # input currents
        input_e = params["w_ee"] * E - params["w_ei"] * I + params["P_e"]
        input_i = params["w_ie"] * E - params["w_ii"] * I + params["P_i"]

        S_e = self._S_e(input_e)
        S_i = self._S_i(input_i)

        dE_dt = (-E + S_e) / tau_e
        dI_dt = (-I + S_i) / tau_i
        return np.array([dE_dt, dI_dt])

    def plot(self, times: np.ndarray, states: np.ndarray, show: bool = True):
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

from typing import Dict, Optional, Tuple
import numpy as np


class BaseModel:
    """
    Minimal base class for time-stepped dynamical systems using Euler integration.

    Subclasses should implement:
        _dynamics(state, t, params) -> np.ndarray

    Attributes:
        params: dict of model parameters
        state: numpy array (state vector)
        dt: time step
        t: current simulation time
    """

    def __init__(self,
                 initial_state: np.ndarray,
                 params: Optional[Dict] = None,
                 dt: float = 0.1):
        self.params = params or {}
        self.state = np.array(initial_state, dtype=float)
        self.dt = float(dt)
        self.t = 0.0

    def reset(self, initial_state: Optional[np.ndarray] = None):
        """Reset time and optionally the state."""
        if initial_state is not None:
            self.state = np.array(initial_state, dtype=float)
        self.t = 0.0

    def _dynamics(self, state: np.ndarray, t: float, params: Dict) -> np.ndarray:
        """Return time derivative (dstate/dt). Must be implemented by subclasses."""
        raise NotImplementedError

    def _step(self):
        """Perform one Euler integration step."""
        deriv = self._dynamics(self.state, self.t, self.params)
        self.state = self.state + self.dt * deriv
        self.t += self.dt

    def simulate(self,
                 duration: float,
                 record_states: bool = True,
                 report_every: Optional[int] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate the model for the given duration using Euler integration.

        Returns:
            times: (N_steps+1,)
            states: (N_steps+1, state_dim)
        """
        n_steps = int(np.ceil(duration / self.dt))
        times = np.zeros(n_steps + 1)
        times[0] = self.t

        if record_states:
            states = np.zeros((n_steps + 1, self.state.size))
            states[0] = self.state.copy()

        for i in range(1, n_steps + 1):
            self._step()
            times[i] = self.t
            if record_states:
                states[i] = self.state.copy()
            if report_every is not None and (i % report_every == 0):
                print(f"t={self.t:.3f}s")

        return (times, states) if record_states else (times, self.state.copy())

    def set_params(self, **kwargs):
        """Update model parameters."""
        self.params.update(kwargs)

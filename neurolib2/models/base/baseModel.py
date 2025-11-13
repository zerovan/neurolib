from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import lax
from dataclasses import dataclass


@dataclass
class BaseModel:
    
    state: jnp.ndarray
    params: Dict[str, float]
    dt: float = 0.1
    t: float = 0.0

    def reset(self, state: Optional[jnp.ndarray] = None):
        if state is not None:
            self.state = state
        self.t = 0.0

    def _dynamics(self, state: jnp.ndarray, t: float, params: Dict[str, float]) -> jnp.ndarray:
        raise NotImplementedError

    def _step(self, state: jnp.ndarray, t: float, params: Dict[str, float]) -> Tuple[jnp.ndarray, float]:    
        deriv = self._dynamics(state, t, params)
        new_state = state + self.dt * deriv
        return new_state, t + self.dt

    def simulate(self, duration: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        n_steps = int(jnp.ceil(duration / self.dt))

        def body(carry, _):
            state, t = carry
            new_state, new_t = self._step(state, t, self.params)
            return (new_state, new_t), (new_t, new_state)

        (_, _), (times, states) = lax.scan(body, (self.state, self.t), None, length=n_steps)
        times = jnp.concatenate([jnp.array([self.t]), times])
        states = jnp.vstack([self.state, states])
        return times, states

    def set_params(self, **kwargs):
        new_params = {**self.params, **kwargs}
        return BaseModel(state=self.state, params=new_params, dt=self.dt, t=self.t)

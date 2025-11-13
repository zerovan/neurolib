"""
wilson_cowan_jax_stable.py
A numerically stable, fully differentiable Wilson–Cowan simulator in JAX
with parameter fitting via gradient descent.

Run:
    python wilson_cowan_jax_stable.py
"""

import jax
import jax.numpy as jnp
from jax import jit, grad
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


# -------------------------
# Wilson–Cowan equations
# -------------------------
def sigmoid(x, gain=1.0, theta=0.0):
    """Numerically stable sigmoid."""
    z = gain * (x - theta)
    z = jnp.clip(z, -60.0, 60.0)  # avoid overflow
    return 1.0 / (1.0 + jnp.exp(-z))


def wc_rhs(state, params):
    """
    Right-hand side of Wilson–Cowan equations.
    state: [E, I]
    params: [wEE, wEI, wIE, wII, PE, PI, tauE, tauI, gain_E, gain_I, theta_E, theta_I]
    """
    E, I = state
    (wEE, wEI, wIE, wII, PE, PI,
     tauE, tauI, gain_E, gain_I, theta_E, theta_I) = params

    input_E = wEE * E - wEI * I + PE
    input_I = wIE * E - wII * I + PI

    S_E = sigmoid(input_E, gain_E, theta_E)
    S_I = sigmoid(input_I, gain_I, theta_I)

    dE = (-E + S_E) / tauE
    dI = (-I + S_I) / tauI
    return jnp.stack([dE, dI])


# -------------------------
# RK4 integration (single step)
# -------------------------
@partial(jit, static_argnums=(0,))
def rk4_step(fun, state, params, dt):
    k1 = fun(state, params)
    k2 = fun(state + 0.5 * dt * k1, params)
    k3 = fun(state + 0.5 * dt * k2, params)
    k4 = fun(state + dt * k3, params)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# -------------------------
# Integrate over time
# -------------------------
@partial(jit, static_argnums=(3,))
def simulate(state0, params, dt, num_steps):
    """Simulate forward dynamics for num_steps."""
    def step_fn(state, _):
        new_state = rk4_step(wc_rhs, state, params, dt)
        return new_state, new_state

    _, traj = jax.lax.scan(step_fn, state0, None, length=num_steps)
    traj = jnp.vstack([state0, traj])
    return traj


# -------------------------
# Utilities
# -------------------------
def make_params(wEE=10.0, wEI=12.0, wIE=10.0, wII=3.0,
                PE=0.5, PI=0.0, tauE=0.02, tauI=0.01,
                gain_E=1.0, gain_I=1.0, theta_E=0.0, theta_I=0.0):
    return jnp.array([wEE, wEI, wIE, wII, PE, PI, tauE, tauI,
                      gain_E, gain_I, theta_E, theta_I])


def mse_loss(pred_traj, target_traj):
    diff = pred_traj - target_traj
    return jnp.mean(diff**2)


@partial(jit, static_argnums=(3,))
def loss_for_params(params, state0, dt, num_steps, target_traj):
    sim = simulate(state0, params, dt, num_steps)
    return mse_loss(sim, target_traj)


grad_loss = partial(jit, static_argnums=(3,))(grad(loss_for_params))


# -------------------------
# Demo: synthetic data and fitting
# -------------------------
def demo_fit():
    # simulation settings
    dt = 0.001          # small step for stability
    T = 20000           # number of steps (total time ≈ 20)
    num_steps = int(T)
    t = np.linspace(0, dt * T, T + 1)

    # true parameters (generate target data)
    params_true = make_params(wEE=10.0, wEI=12.0, wIE=10.0, wII=3.0,
                              PE=0.8, PI=0.2, tauE=0.02, tauI=0.01,
                              gain_E=1.5, gain_I=1.3)

    # initial state
    state0 = jnp.array([0.1, 0.1])

    # generate noisy "experimental" data
    traj_true = simulate(state0, params_true, dt, num_steps)
    rng = np.random.default_rng(0)
    traj_noisy = np.array(traj_true) + rng.normal(scale=0.02, size=traj_true.shape)
    target_traj = jnp.array(traj_noisy)

    # initial guess
    params_init = make_params(wEE=8.0, wEI=8.0, wIE=9.0, wII=2.5,
                              PE=0.4, PI=0.1, tauE=0.02, tauI=0.01,
                              gain_E=1.0, gain_I=1.0)

    # indices to optimize (subset)
    # opt_idx = jnp.arange(len(params_init))
    opt_idx = jnp.array([0, 1, 4, 5])  # wEE, wEI, PE, PI

    params = params_init.copy()
    lr = 0.1  # smaller for safety
    n_steps = 2000

    print("Starting parameter fitting...")
    for i in range(n_steps):
        g = grad_loss(params, state0, dt, num_steps, target_traj)
        update = jnp.zeros_like(params).at[opt_idx].set(g[opt_idx])
        params = params - lr * update
        params = jnp.clip(params, -20.0, 20.0)

        if i % 20 == 0:
            loss_val = loss_for_params(params, state0, dt, num_steps, target_traj)
            print(f"Step {i:03d} | loss={float(loss_val):.6f}")

    fitted_traj = simulate(state0, params, dt, num_steps)

    
    # plotting
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, np.array(target_traj)[:, 0], label="target E (noisy)")
    plt.plot(t, np.array(fitted_traj)[:, 0], label="fitted E")
    plt.xlabel("time")
    plt.legend()
    plt.title("Excitatory population")

    plt.subplot(1, 2, 2)
    plt.plot(t, np.array(target_traj)[:, 1], label="target I (noisy)")
    plt.plot(t, np.array(fitted_traj)[:, 1], label="fitted I")
    plt.xlabel("time")
    plt.legend()
    plt.title("Inhibitory population")

    plt.tight_layout()
    plt.show()

    print("\nTrue params (selected):", np.array(params_true)[opt_idx])
    print("Init params (selected):", np.array(params_init)[opt_idx])
    print("Fitted params (selected):", np.array(params)[opt_idx])


if __name__ == "__main__":
    demo_fit()

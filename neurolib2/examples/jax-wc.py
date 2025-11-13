import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import jit, grad, vmap, random, lax

# Wilson-Cowan parameters container (example)
params = {
    "w_ee": 16.0,
    "w_ei": 12.0,
    "w_ie": 15.0,
    "w_ii": 3.0,
    "P": 1.0,   # external drive to E
    "Q": 0.0,   # external drive to I
    "tau_e": 0.1,
    "tau_i": 0.3,
    "beta_e": 1.0,  # sigmoid slope
    "beta_i": 1.0,  # sigmoid slope
    "theta_e": 1.5,  # sigmoid threshold
    "theta_i": 2.0  # sigmoid threshold
}

def S(x, beta=1.0, theta=0.0):
    # smooth sigmoid
    return 1.0 / (1.0 + jnp.exp(-beta * (x - theta)))

def wc_rhs(u, params):
    # u = [E, I]
    E, I = u
    wee = params["w_ee"]; wei = params["w_ei"]
    wie = params["w_ie"]; wii = params["w_ii"]
    P = params["P"]; Q = params["Q"]
    tau_e = params["tau_e"]; tau_i = params["tau_i"]
    beta_e = params["beta_e"]; beta_i = params["beta_i"]
    theta_e = params["theta_e"]; theta_i = params["theta_i"]

    inpE = wee * E - wei * I + P
    inpI = wie * E - wii * I + Q

    dE = (-E + S(inpE, beta_e, theta_e)) / tau_e
    dI = (-I + S(inpI, beta_i, theta_i)) / tau_i
    return jnp.array([dE, dI])

# Explicit Euler step (simple)
@jit
def euler_step(u, dt, params):
    return u + dt * wc_rhs(u, params)

# Full simulation using lax.scan for efficiency
def simulate(step_fn, u0, params, dt=0.01, n_steps=1000):
    def body(carry, _):
        u = carry
        u_next = step_fn(u, dt, params)
        return u_next, u_next  # carry, to-record
    _, traj = lax.scan(body, u0, None, length=n_steps)
    # traj shape: (n_steps, 2)
    return traj

# Example usage
u0 = jnp.array([0.1, 0.1])  # initial E, I
traj = simulate(euler_step, u0, params, dt=0.01, n_steps=1000)



E, I = traj.T
plt.plot(E, label="E (Excitatory)")
plt.plot(I, label="I (Inhibitory)")
plt.legend()
plt.xlabel("Time step")
plt.ylabel("Activity")
plt.title("Wilson–Cowan Simulation")
plt.show()
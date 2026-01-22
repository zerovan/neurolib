import jax
import jax.numpy as jnp
import diffrax


# -----------------------------
# Parameters
# -----------------------------
t0 = 0.0
t1 = 1.0
dt0 = 0.01
sigma = 0.5
key = jax.random.key(34)


# -----------------------------
# Brownian control
# -----------------------------
brownian = diffrax.VirtualBrownianTree(
    t0=t0,
    t1=t1,
    tol=dt0 / 2,
    shape=(),          # scalar Brownian motion
    key=key
)


# -----------------------------
# Diffusion term g(y)
# -----------------------------
def diffusion(t, y, args):
    return sigma

def drift(t, y, args):
    mu = 1.0
    return mu * y

drift_term = diffrax.ODETerm(drift)
diffusion_term = diffrax.ControlTerm(diffusion, brownian)

term = diffrax.MultiTerm(drift_term, diffusion_term)
# -----------------------------
# SDE definition
# -----------------------------
# term = diffrax.ControlTerm(diffusion, brownian)
solver = diffrax.Euler()


# -----------------------------
# Solve SDE
# -----------------------------
sol = diffrax.diffeqsolve(
    term,
    solver,
    t0=t0,
    t1=t1,
    dt0=dt0,
    y0=0.0,
    saveat=diffrax.SaveAt(ts=jnp.linspace(t0, t1, 200))
)


# -----------------------------
# Result
# -----------------------------
import matplotlib.pyplot as plt

plt.plot(sol.ts, sol.ys)
plt.xlabel("time")
plt.ylabel("Y_t")
plt.title("Brownian-driven SDE")
plt.show()

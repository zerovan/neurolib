# Ornstein-Uhlenbeck process test and plot in one file

import jax
import jax.numpy as jnp
import diffrax
import lineax
import matplotlib.pyplot as plt


class OUModel:
    def __init__(self, number_of_regions, key):
        self.number_of_regions = number_of_regions
        self.key = key

    def noise_term(self, ts, tau, mean, sigma):
        def drift(t, y, args, *, history=None):
            return -tau * (y - mean)

        def diffusion(t, y, args, *, history=None):
            return lineax.DiagonalLinearOperator(sigma)

        brownian_motion = diffrax.VirtualBrownianTree(
            ts[0], ts[-1], tol=1e-3, shape=(self.number_of_regions,), key=self.key
        )

        return diffrax.MultiTerm(
            diffrax.ODETerm(drift),
            diffrax.ControlTerm(diffusion, brownian_motion),
        )


# -----------------------------
# Simulation parameters
# -----------------------------
key = jax.random.PRNGKey(0)
model = OUModel(number_of_regions=1, key=key)

tau = 2.0
mean = 1.5
sigma = jnp.array([0.4])

ts = jnp.linspace(0.0, 20.0, 1000)
y0 = jnp.array([5.0])

term = model.noise_term(ts, tau, mean, sigma)

sol = diffrax.diffeqsolve(
    term,
    solver=diffrax.EulerHeun(),
    t0=ts[0],
    t1=ts[-1],
    dt0=1e-2,
    y0=y0,
    saveat=diffrax.SaveAt(ts=ts),
)

# -----------------------------
# Plot
# -----------------------------
plt.figure()
plt.plot(ts, sol.ys[:, 0])
plt.axhline(mean)
plt.xlabel("Time")
plt.ylabel("State")
plt.title("Ornstein–Uhlenbeck Process")
plt.show()



# -----------------------------
# Setup
# -----------------------------
key = jax.random.PRNGKey(0)

t0 = 0.0
t1 = 10.0
ts = jnp.linspace(t0, t1, 1000)

dim = 30  # number of regions / dimensions

# Create VirtualBrownianTree
bm = diffrax.VirtualBrownianTree(
    t0=t0,
    t1=t1,
    tol=1e-3,
    shape=(dim,),
    key=key,
)

# Evaluate Brownian motion at times ts
W = jax.vmap(bm.evaluate)(ts)

# -----------------------------
# Plot
# -----------------------------
plt.figure()
for i in range(dim):
    plt.plot(ts, W[:, i], label=f"Dimension {i}")

plt.xlabel("Time")
plt.ylabel("Brownian value")
plt.title("VirtualBrownianTree paths")
plt.legend()
plt.show()

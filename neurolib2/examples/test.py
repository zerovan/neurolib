import jax
import jax.numpy as jnp
import diffrax
import matplotlib.pyplot as plt

def logistic(t, y, args):
    alpha = args
    return alpha * y * (1 - y)

y0 = 0.1
t0, t1 = 0.0, 10.0
ts = jnp.linspace(t0, t1, 200)

solver = diffrax.Dopri5()
saveat = diffrax.SaveAt(ts=ts)

sol = diffrax.diffeqsolve(
    diffrax.ODETerm(logistic),
    solver,
    t0=t0,
    t1=t1,
    dt0=0.01,
    y0=y0,
    args=1.0,
    saveat=saveat
)

plt.plot(sol.ts, sol.ys)
plt.show()

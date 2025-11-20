import time
import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

def delay_logistic_vf(t, y, args, *, history):
    alpha = args
    return alpha * y * (1 - history[0])


def history_function(t):
    return 2.0

delays = diffrax.Delays(
    delays=[lambda t, y, args: 1.0], initial_discontinuities=jnp.array([0.0])
)

@jax.jit
def main(alpha):
    terms = diffrax.ODETerm(delay_logistic_vf)
    t0 = 0.0
    t1 = 20.0
    ts = jnp.linspace(0, 20, 200)
    solver = diffrax.Bosh3()
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-6)
    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0,
        t1,
        ts[1] - ts[0],
        y0=history_function,
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        stepsize_controller=stepsize_controller,
        delays=delays,
        args=alpha,
    )
    return sol

alpha = 2.0
main(alpha)

start = time.time()
sol = main(alpha)
end = time.time()
print(f"Integration took in {end - start} seconds.")

plt.plot(sol.ts, sol.ys)
plt.show()

main(1)

start = time.time()
sol = main(1)
end = time.time()
print(f"Integration took in {end - start} seconds.")

plt.plot(sol.ts, sol.ys)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

def DDEs(number, tau, x, h, k):
    # Euler steps for the DDE
    for i in range(1, number):
        ti = t[i]
        if ti - tau < 0:
            # use history function
            x_delay = history(ti - tau)
        else:
            # interpolate between known points
            delay_index = (ti - tau) / h
            i0 = int(np.floor(delay_index))
            alpha = delay_index - i0
            if i0 + 1 < len(x):
                x_delay = (1 - alpha) * x[i0] + alpha * x[i0 + 1]
            else:
                x_delay = x[i0]
        x[i] = x[i - 1] + h * (-k * x_delay)
    return x


# Parameters
k = 1.0
taus = np.arange(1.1, 1.7, 0.5)
h = 0.01         # step size
T = 20           # total time

# Time vector
t = np.arange(0, T + h, h)
N = len(t)

# History function for t in [-tau,0]
def history(t):
    return 1.0  # constant history

# --- Solve ODE: x'(t) = -k x(t) ---
x_ode = np.zeros(N)
x_ode[0] = history(0)

for i in range(N - 1):
    x_ode[i + 1] = x_ode[i] + h * (-k * x_ode[i])

# --- Solve DDE: x'(t) = -k x(t - tau) using Euler + interpolation ---
x_dde = np.zeros((N, len(taus)))

# Fill initial history segment
for i, ti in enumerate(t):
    if ti <= 0:
        x_dde[i] = history(ti)
    else:
        break
    


for j in range(len(taus)):
    x_dde[:,j] = DDEs(N,taus[j], x_dde[:,j],h, k)
    

# --- Plot ---
plt.figure(figsize=(8, 5))
plt.plot(t, x_ode, label="ODE: no delay")
for i in range(len(taus)):
    plt.plot(t, x_dde[:,i], label=f"DDE: delay = {taus[i]}")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.legend()
plt.title("Effect of Delay in a Simple DDE System")
plt.show()

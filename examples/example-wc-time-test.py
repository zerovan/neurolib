from neurolib.utils.loadData import Dataset
from neurolib.models.wc import WCModel
import matplotlib.pyplot as plt
import time
import numpy as np
# load structural data
ds = Dataset("hcp")
print("Cmat shape:", ds.Cmat.shape, "Dmat shape:", ds.Dmat.shape)

# create Neurolib-1 model with same connectivity
model = WCModel(Cmat=ds.Cmat, Dmat=ds.Dmat, seed=0)

model.params['exc_ext'] = 0.65
model.params['signalV'] = 0
model.params['sigma_ou'] = 0.14
model.params['K_gl'] = 3.15

# set duration in ms (Neurolib-1 uses ms)
duration_seconds = 500.0
model.params["duration"] = duration_seconds #* 1000.0  # 10 s -> 10000 ms

# optional: make dt explicit to match Neurolib-2 settings
model.params["dt"] = 0.01  # ms step (if you used 0.01 in neurolib2, be consistent)

# Warm-up run (not timed) — helps remove IO/initialization overhead
print("Warm-up run ...")
model.run()
# clear or reset any stored state if necessary (Neurolib-1 stores time series in model.exc etc.)

# Now timed run
print("Timed run ...")
t0 = time.perf_counter()
model.run()
t1 = time.perf_counter()

elapsed = t1 - t0
print(f"Neurolib-1 run took {elapsed:.3f} seconds")
print("exc shape:", getattr(model, "exc", None).shape)

print("exc shape:", model.exc.shape)
print("number of regions:", model.exc.shape[0])



# Convert Neurolib1 outputs
E = model.exc   # (N, T)
I = model.inh   # (N, T)

states = np.stack([E, I], axis=0)        # (2, N, T)
states = np.transpose(states, (2, 0, 1)) # (T, 2, N)

times = model.t / 1000.0   # convert ms → seconds

num_regions = states.shape[2]

plt.figure(figsize=(10,5))

for r in range(2):
    plt.plot(times, states[:,0,r], label=f"E node {r}")
    plt.plot(times, states[:,1,r], "--", label=f"I node {r}")

plt.xlabel("time (s)")
plt.ylabel("activity")
plt.title("Wilson-Cowan: all node activities (Neurolib1)")
plt.legend()
plt.tight_layout()

plt.show()
import os, sys
import jax
jax.config.update("jax_disable_jit", False)
jax.config.update("jax_enable_compilation_cache", True)
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from neurolib.models.wc import WCModel
from neurolib.utils.loadData import Dataset

# First we load the structural data set from the Human Connectome Project 
ds = Dataset("hcp")

# We initiate the Wilson-Cowan model
# wc = WCModel(Cmat = ds.Cmat, Dmat = ds.Dmat, seed=0)
print("fiber count matrix shape:", ds.Cmat.shape)
print("fiber length matrix shape:", ds.Dmat.shape)


from neurolib2.models.wilsonCowanModel.wilsonCowanModel import WilsonCowan

# Example: run simulation when module executed as script
if __name__ == "__main__":
    # model = WilsonCowan.create_default(dt=0.01)
    model = WilsonCowan.create_default(
        dt=0.01,
        fiber_length_matrix=ds.Dmat,
        fiber_count_matrix=ds.Cmat,
    )

    start = time.time()

    times, states = model.simulate(500)
    model.plot(times, states)
    states.block_until_ready()   # IMPORTANT for correct timing

    end = time.time()

    print(f"Run took {end - start:.2f} seconds")
    print("States shape:", states.shape)

    # grad_model = model.derivative_model(10.0, with_respect_to=["tau_e", "w_ee"])
    # print(grad_model.tau_e, grad_model.w_ee)

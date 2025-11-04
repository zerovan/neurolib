import matplotlib.pyplot as plt
import numpy as np
# import glob

from neurolib.models.wc import WCModel

# import neurolib.utils.loadData as ld
# import neurolib.utils.functions as func

model = WCModel()
model.params['duration'] = 2.0*1000

max_exc = []
min_exc = []
# these are the different input values that we want to scan
exc_inputs = np.linspace(0, 3.5, 50)
for exc_ext in exc_inputs:
    # Note: this has to be a vector since it is input for all nodes
    # (but we have only one node in this example)
    model.params['exc_ext'] = exc_ext
    model.run()
    # we add the maximum and the minimum of the last second of the 
    # simulation to a list
    max_exc.append(np.max(model.exc[0, -int(1000/model.params['dt']):]))
    min_exc.append(np.min(model.exc[0, -int(1000/model.params['dt']):]))
    
plt.plot(exc_inputs, max_exc, c='k', lw = 2)
plt.plot(exc_inputs, min_exc, c='k', lw = 2)
plt.title("Bifurcation diagram of the Wilson-Cowan model")
plt.xlabel("Input to exc")
plt.ylabel("Min / max exc")
plt.show()
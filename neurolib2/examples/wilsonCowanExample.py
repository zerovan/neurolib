import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from neurolib2.models.wilsonCowanModel.wilsonCowanModel import WilsonCowan

# Example: run simulation when module executed as script
if __name__ == "__main__":
    model = WilsonCowan.create_default(dt=0.01)
    times, states = model.simulate(2)
    model.plot(times, states)
    print(states)
    # grad_model = model.derivative_model(10.0, with_respect_to=["tau_e", "w_ee"])
    # print(grad_model.tau_e, grad_model.w_ee)

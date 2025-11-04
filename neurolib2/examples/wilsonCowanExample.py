
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from neurolib2.models.wilsonCowanModel.wilsonCowanModel import WilsonCowan


# Example: run simulation when module executed as script
if __name__ == "__main__":
    model = WilsonCowan(dt=0.05)
    # small tonic input to E
    model.set_params(P_e=1.0, P_i=0.0)
    times, states = model.simulate(duration=1000.0)
    model.plot(times, states)
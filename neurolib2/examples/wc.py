import numpy as np
import matplotlib.pyplot as plt

class wc:
    def __init__(self, ):
      self.Tau_e = 10
      self.Tau_i = 20
      self.w_ee = 16
      self.w_ei = 12
      self.w_ii = 3
      self.w_ie = 15
      self.p_e = 1
      self.p_i = 0
      self.a_e = 1
      self.a_i = 1
      self.theta_e = 1.5
      self.theta_i = 2
      
      self.dt = 0.1
      self.duration = 1000
    
    def dynamics(self, state):
        
        def sigmoidE(x): 
            return 1 / (1 + np.exp(-self.a_e * (x - self.theta_e)))
        def sigmoidI(x): 
            return 1 / (1 + np.exp(-self.a_i * (x - self.theta_i)))
        
        e = state[0]
        i = state[1]
        
        dStateE = (-e + sigmoidE(self.w_ee * e - self.w_ei * i + self.p_e)) / self.Tau_e
        dStateI = (-i + sigmoidI(self.w_ie * e - self.w_ii * i + self.p_i)) / self.Tau_i
        
        return np.array([dStateE, dStateI])
    
    def steps(self,):
        self.steps = int(self.duration / self.dt)
        self.states = np.zeros((self.steps,2))
        
        for i in range(0, self.steps-1):
            state = self.dynamics(self.states[i])
            self.states[i+1] = self.states[i] + (self.dt * state)
            
        self.plot()

    def plot(self):
        print(self.states[:,0])
        print(self.states[:,1])
        plt.plot( np.arange(self.steps), self.states[:,0],np.arange(self.steps), self.states[:,1])
        
        plt.show()

wilsonCowan = wc()
wilsonCowan.steps()
    
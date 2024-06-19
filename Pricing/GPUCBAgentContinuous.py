import numpy as np
from Pricing.RBFGaussianProcess import RBFGaussianProcess
import scipy

class GPUCBAgentContinuous:
    def __init__(self, T):
        self.T = T
        self.gp = RBFGaussianProcess(scale=2).fit()
        self.a_t = None
        self.action_hist = np.array([])
        self.reward_hist = np.array([])
        self.mu_t = 0
        self.sigma_t = 0 #1
        self.gamma = lambda t: np.log(t+1)**2 
        self.beta = lambda t: 1 + 0.5*np.sqrt(2 * (self.gamma(t) + 1 + np.log(T)))
        self.t = 0
    
    def pull_arm(self):
        self.a_t = scipy.optimize.minimize(self.ucbs_func, x0 = 0.5, bounds = [(0,1)]).x
        return self.a_t[0]
    
    def ucbs_func(self, price):
        self.mu_t, self.sigma_t = self.gp.predict(price) 
        return 1/(self.mu_t + self.beta(self.t) * self.sigma_t)
    
    def update(self, r_t):
        self.action_hist = np.append(self.action_hist, self.a_t)
        self.reward_hist = np.append(self.reward_hist, r_t)
        self.gp = self.gp.fit(self.a_t, r_t)
        self.t += 1
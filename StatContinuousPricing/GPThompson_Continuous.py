import numpy as np
from RBFGaussianProcess import RBFGaussianProcess
import scipy

class GPThompson:
    def __init__(self, T):
        self.T = T
        self.gp = RBFGaussianProcess(scale=2).fit()
        self.a_t = None
        self.action_hist = np.array([])
        self.reward_hist = np.array([])
        self.mu_t = 0
        self.sigma_t = 0
        self.gamma = lambda t: np.log(t+1)**2 
        self.beta = lambda t: 1 + 0.5*np.sqrt(2 * (self.gamma(t) + 1 + np.log(T)))
        self.t = 0
    
    def pull_arm(self):
        self.a_t = scipy.optimize.minimize(self.obj, x0 = 0.5, bounds = [(0,1)]).x
        return self.a_t
    
    def obj(self, arm):
        self.mu_t, self.sigma_t = self.gp.predict(arm) 
        sample = np.random.normal(self.mu_t, self.sigma_t)
        return 1/sample
    
    def update(self, r_t):
        self.action_hist = np.append(self.action_hist, self.a_t)
        self.reward_hist = np.append(self.reward_hist, r_t)
        self.gp = self.gp.fit(self.a_t, r_t)
        self.t += 1
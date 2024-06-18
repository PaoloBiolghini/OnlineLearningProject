import numpy as np
class MultiplicativePacingAgent:
    def __init__(self, valuation, budget, T, n_users, eta):
        self.valuation = valuation
        self.budget = budget
        self.eta = eta
        self.T = T
        self.n_users = n_users
        self.rho = self.budget/(self.T*self.n_users)
        self.lmbd = 1
        self.t = 0

    def bid(self):
        if self.budget < 1:
            return 0
        return self.valuation/(self.lmbd+1)
    
    def update(self, f_t, c_t):
        self.lmbd = np.clip(self.lmbd-self.eta*(self.rho-c_t), 
                            a_min=0, a_max=1/self.rho)
        self.budget -= c_t
    def update_per_round(self):
        self.t += 1
        if self.t < self.T:
            self.rho = self.budget/((self.T-self.t)*self.n_users)
        else:
            self.rho = self.budget/(self.n_users)
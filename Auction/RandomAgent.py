from Auction.Agent import Agent
import numpy as np

class RandomAgent(Agent):
    def __init__(self, valuation, budget, T, eta):
        self.valuation = valuation
        self.budget = budget
        self.eta = eta
        self.T = T
        self.t=0

    def bid(self):
       return np.random.uniform(0, 1) 
    
    def update(self, f_t, c_t):
        pass
    def update_valuation(self):
        pass
import numpy as np 
import scipy

class NonstationaryAdvEnvironment:
    def __init__(self, T, n_users, seed, n_arms):
        np.random.seed(seed)
        
        # Pricing Env
        self.t = 0 
        self.prices = np.linspace(0,1,n_arms) # price discretization
        user_effect = np.random.normal(0,0.2,size = (n_users, T)) # user specific effect for every day 
        
        prob_func = lambda p,t: scipy.special.expit(5*p*t/T) # design choice
        self.prob_history = np.vectorize(prob_func)(self.prices[:, np.newaxis], range(1,T+1))
        #creates a matrix where each row corresponds to a price and each column corresponds to a time step, 
        #storing the calculated probability for each combination.
        self.current_prob = np.zeros(n_users)
        
        
    def get_prob(self, a_t):
        self.current_prob = self.prob_history[a_t, self.t] + np.random.normal(0, 0.2, self.current_prob.shape)
        self.current_prob = np.maximum(0, np.minimum(1, self.current_prob))
        self.t += 1
        return self.current_prob
    
    
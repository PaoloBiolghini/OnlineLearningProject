import numpy as np 
class PricingSlightlyNonstationaryEnvironment:
    def __init__(self, conversion_probability, cost, variance):
        self.conversion_probability = conversion_probability
        self.cost = cost
        self.variance=variance
        self.t=0

    def round(self, p_t, n_t):
        "n_t: number of users, p_t: price"
        #get the probability and clip it between 0 and 1
        probability=self.conversion_probability(p_t, self.t)+np.random.normal(0,self.variance)
        probability=np.clip(probability,0,1)
        #get the demand
        d_t = np.random.binomial(n_t,probability )
        r_t = (p_t - self.cost)*d_t
        self.t+=1
        "return: demand, profit"
        return d_t, r_t 
    
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class StochasticPricingEnvironment:
    def __init__(self, conversion_probability, cost):
        self.conversion_probability = conversion_probability
        self.cost = cost
        self.t = 0
        
    def round(self, p_t, n_t):
        d_t = np.random.binomial(n_t, self.conversion_probability(p_t, self.t))
        r_t = (p_t - self.cost)*d_t
        self.t += 1
        return d_t, r_t


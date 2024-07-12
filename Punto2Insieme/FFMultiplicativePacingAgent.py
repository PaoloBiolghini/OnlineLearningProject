#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[3]:


class Hedge:  # Hedge Regret-Minimizer
    def __init__(self, K, learning_rate):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.x_t = np.ones(K) / K
        self.a_t = None
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights / sum(self.weights)
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.a_t

    def update(self, l_t):
        self.weights *= np.exp(-self.learning_rate * l_t)
        self.t += 1


# In[4]:


class FFMultiplicativePacingAgent:
    def __init__(self, K, valuation, budget, T, eta):
        self.K = K
        self.bids_set = np.linspace(0, 1, self.K)
        self.hedge = Hedge(self.K, np.sqrt(np.log(self.K) / T))
        self.valuation = valuation
        self.budget = budget
        self.eta = eta
        self.T = T
        self.rho = self.budget / self.T
        self.lmbd = 1
        self.t = 0

    def bid(self):
        if self.budget < 1:
            return 0
        return self.bids_set[self.hedge.pull_arm()]

    def update(self, f_t, c_t, m_t):
        f_t_full = np.array([(self.valuation - b) * int(b >= m_t) for b in self.bids_set])  # bidder utility
        c_t_full = np.array([b * int(b >= m_t) for b in self.bids_set])  # bidder costs
        L = f_t_full - self.lmbd * (c_t_full - self.rho)  # lagrangian
        range_L = 2 + (1 - self.rho) / self.rho
        self.hedge.update((2 - L) / range_L)

        # primal-dual update
        self.lmbd = np.clip(self.lmbd - self.eta * (self.rho - c_t),
                            a_min=0, a_max=1 / self.rho)
        # budjet update
        self.budget -= c_t
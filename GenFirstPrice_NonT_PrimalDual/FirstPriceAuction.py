#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class FirstPriceAuction:
    def __init__(self, ctrs):
        self.ctrs = ctrs
        self.n_adv = len(self.ctrs)
    
    def get_winners(self, bids):
        adv_values = self.ctrs*bids
        adv_ranking = np.argsort(adv_values)
        winner = adv_ranking[-1]
        return winner, adv_values
    
    def get_payments_per_click(self, winners, values, bids):
        payment = bids[winners]
        return payment.round(2)
    
    def round(self, bids):
        winners, values = self.get_winners(bids) 
        payments_per_click = self.get_payments_per_click(winners, values, bids)
        return winners, payments_per_click


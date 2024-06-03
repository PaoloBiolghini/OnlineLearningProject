import numpy as np 

class NonstationaryAdvEnvironment:
    def __init__(self, T, n_competitors, auctions_per_day, seed):
        np.random.seed(seed)
                    
        
        # Auction Env
        self.auc_t = 0 
        self.n_auctions = auctions_per_day*T
        
        sin = np.sin(np.linspace(0,10, self.n_auctions)) # design choice
        self.bids_history = np.tile(sin, (n_competitors,1)) + np.random.normal(0,1,size=(n_competitors,self.n_auctions))
        self.bids_history = np.maximum(0,self.bids_history)
        
        self.current_reward = 0
        self.cumulative_reward = 0
        
    
    def get_bid(self):
        bid = self.bids_history[:,self.auc_t]
        self.auc_t += 1
        return bid
    
    def next_round(self, slots_won, a_t):
        prob = self.current_prob 
        self.current_reward = slots_won * np.dot(prob, np.ones(prob.shape) * self.prices[a_t])
        self.cumulative_reward += self.current_reward
        pass
import numpy as np

class EXP3PAgent:
    def __init__(self, K, learning_rate, gamma, T):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)* ((learning_rate*gamma/3) * np.sqrt(T/K) )
        self.a_t = None
        self.x_t = np.ones(K)/K
        self.N_pulls = np.zeros(K)
        self.t = 0
        self.gamma = gamma
        self.T= T

    def pull_arm(self):
        self.x_t = (1-self.gamma)*self.weights/sum(self.weights) + self.gamma/self.K
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.a_t
    
    def update(self, l_t):
        l_t_tilde = l_t/self.x_t[self.a_t]
        self.weights[self.a_t] *= np.exp( self.gamma/(3*self.K)*(l_t_tilde+ self.learning_rate/(self.x_t[self.a_t]*np.sqrt(self.K* self.T))))
        self.N_pulls[self.a_t] += 1
        self.t += 1
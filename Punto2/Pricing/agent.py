import numpy as np

#---------EXP3--------------
class EXP3Agent:
    def __init__(self, K, learning_rate):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.a_t = None
        self.x_t = np.ones(K)/K
        self.N_pulls = np.zeros(K)
        self.t = 0
        self.original_params = (K, learning_rate)

    def pull_arm(self):
        self.x_t = self.weights/sum(self.weights)
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.a_t
    
    def update(self, l_t):
        l_t_tilde = l_t/self.x_t[self.a_t]
        self.weights[self.a_t] *= np.exp(-self.learning_rate*l_t_tilde)
        self.N_pulls[self.a_t] += 1
        self.t += 1

    def reset(self):
        self.__init__(*self.original_params)


#--------EXP3 sliding window-------------
class EXP3AgentSlidingWindow:
    def __init__(self, K, learning_rate, W):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.a_t = None
        self.x_t = np.ones(K) / K
        self.N_pulls = np.zeros(K)
        self.t = 0
        self.loss_queue=[]
        self.refArm_queue=[]
        self.W=W
        self.original_params=(K, learning_rate, W)

    def pull_arm(self):
        self.x_t = self.weights / sum(self.weights)
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.a_t

    def update(self, l_t):
        l_t_tilde = l_t / self.x_t[self.a_t]
        self.loss_queue.append(l_t_tilde)
        self.refArm_queue.append(self.a_t)
        if(self.t>self.W):
            queueLoss=self.loss_queue.pop(0)
            armtoModify=self.refArm_queue.pop(0)
            self.weights[armtoModify]*= np.exp(self.learning_rate * queueLoss)

        self.weights[self.a_t] *= np.exp(-self.learning_rate * l_t_tilde)
        self.N_pulls[self.a_t] += 1
        self.t += 1

    def reset(self):
        self.__init__(*self.original_params)


#-----------EXP3.P-----------
import numpy as np

class EXP3PAgent:
    def __init__(self, K, gamma, beta, eta):
        self.K = K
        self.gamma = gamma
        self.beta = beta
        self.eta = eta
        self.weights = np.ones(K)
        self.probabilities = np.ones(K) / K
        self.losses = np.zeros(K)
        self.t = 0
        self.original_params = (K, gamma, beta, eta)

    def pull_arm(self):
        self.probabilities = (1 - self.gamma) * (self.weights / np.sum(self.weights)) + self.gamma / self.K
        self.a_t = np.random.choice(np.arange(self.K), p=self.probabilities)
        return self.a_t

    def update(self, reward):
        # Update the estimated loss for the chosen arm
        estimated_loss = (1 - reward) / self.probabilities[self.a_t]
        self.losses[self.a_t] += estimated_loss
        
        # Update the weights
        self.weights[self.a_t] *= np.exp(-self.eta * estimated_loss)
        
        # Perturb the weights
        noise = np.random.normal(0, 1, self.K)
        self.weights *= np.exp(self.beta * noise)
        
        self.t += 1

    def reset(self):
        self.__init__(*self.original_params)

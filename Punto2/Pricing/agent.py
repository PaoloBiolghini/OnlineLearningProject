import numpy as np

#EXP3
class EXP3Agent:
    def __init__(self, K, learning_rate):
        self.K = K
        self.learning_rate = learning_rate
        self.weights = np.ones(K)
        self.a_t = None
        self.x_t = np.ones(K)/K
        self.N_pulls = np.zeros(K)
        self.t = 0

    def pull_arm(self):
        self.x_t = self.weights/sum(self.weights)
        self.a_t = np.random.choice(np.arange(self.K), p=self.x_t)
        return self.a_t
    
    def update(self, l_t):
        l_t_tilde = l_t/self.x_t[self.a_t]
        self.weights[self.a_t] *= np.exp(-self.learning_rate*l_t_tilde)
        self.N_pulls[self.a_t] += 1
        self.t += 1


#EXP3 sliding window
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


#EXP3.P
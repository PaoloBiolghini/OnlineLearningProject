import numpy as np
#UCB classical
class UCB1Agent:
    def __init__(self, K, T, range=1):
        self.K = K
        self.T = T
        self.range = range
        self.a_t = None
        self.average_rewards = np.zeros(K)
        self.N_pulls = np.zeros(K)
        self.t = 0
        self._original_params = (K, T, range)  # Store the original parameters

    def pull_arm(self):
        if self.t < self.K:
            self.a_t = self.t
        else:
            ucbs = self.average_rewards + self.range * np.sqrt(2 * np.log(self.T) / self.N_pulls)
            self.a_t = np.argmax(ucbs)
        return self.a_t

    def update(self, r_t):
        self.N_pulls[self.a_t] += 1
        self.average_rewards[self.a_t] += (r_t - self.average_rewards[self.a_t]) / self.N_pulls[self.a_t]
        self.t += 1

    def reset(self):
        self.__init__(*self._original_params)  # Reinitialize with the original parameters

#UCB SW
class SWUCBAgent:
    def __init__(self, K, T, W, range=1):
        self.K = K
        self.T = T
        self.W = W
        self.range = range
        self.a_t = None
        self.cache = np.repeat(np.nan, repeats=K * W).reshape(W, K)
        self.N_pulls = np.zeros(K)
        self.t = 0
        self._original_params = (K, T, W, range)  # Store the original parameters

    def pull_arm(self):
        if self.t < self.K:
            self.a_t = self.t
        else:
            n_pulls_last_w = self.W - np.isnan(self.cache).sum(axis=0)
            avg_last_w = np.nanmean(self.cache, axis=0)
            ucbs = avg_last_w + self.range * np.sqrt(2 * np.log(self.W) / n_pulls_last_w)
            self.a_t = np.argmax(ucbs)
        return self.a_t

    def update(self, r_t):
        self.N_pulls[self.a_t] += 1
        self.cache = np.delete(self.cache, (0), axis=0)  # remove oldest observation
        new_samples = np.repeat(np.nan, self.K)
        new_samples[self.a_t] = r_t
        self.cache = np.vstack((self.cache, new_samples))  # add new observation
        self.t += 1

    def reset(self):
        self.__init__(*self._original_params)

#--------------UCB CumSum----------------------
class CUSUMUCBAgent:
    def __init__(self, K, T, M, h, alpha=0.99, range=1):
        self.K = K
        self.T = T
        self.M = M
        self.h = h
        self.alpha = alpha
        self.range = range
        self.a_t = None
        self.reset_times = np.zeros(K)
        self.N_pulls = np.zeros(K)
        self.all_rewards = [[] for _ in np.arange(K)]
        self.counters = np.repeat(M, K)
        self.average_rewards = np.zeros(K)
        self.n_resets = np.zeros(K)
        self.n_t = 0
        self.t = 0
        self.count=0
        self._original_params = (K, T, M, h, alpha, range)

    def pull_arm(self):
        if (self.counters > 0).any():
            for a in np.arange(self.K):
                if self.counters[a] > 0:
                    self.counters[a] -= 1
                    self.a_t = a
                    break
        else:
            if np.random.random() <= 1 - self.alpha:
                ucbs = self.average_rewards + self.range * np.sqrt(np.log(self.n_t) / self.N_pulls)
                self.a_t = np.argmax(ucbs)
            else:
                self.a_t = np.random.choice(np.arange(self.K))  # extra exploration
        return self.a_t

    def update(self, r_t):
        self.N_pulls[self.a_t] += 1
        self.all_rewards[self.a_t].append(r_t)
        if self.counters[self.a_t] == 0:
            if self.change_detection():
                self.n_resets[self.a_t] += 1
                self.N_pulls[self.a_t] = 0
                self.average_rewards[self.a_t] = 0
                self.counters[self.a_t] = self.M
                self.all_rewards[self.a_t] = []
                self.reset_times[self.a_t] = self.t
            else:
                self.average_rewards[self.a_t] += (r_t - self.average_rewards[self.a_t]) / self.N_pulls[self.a_t]
        self.n_t = sum(self.N_pulls)
        self.t += 1

    def change_detection(self):
        ''' CUSUM CD sub-routine. This function returns 1 if there's evidence that the last pulled arm has its average reward changed '''
        u_0 = np.mean(self.all_rewards[self.a_t][:self.M])
        sp, sm = (
        np.array(self.all_rewards[self.a_t][self.M:]) - u_0, u_0 - np.array(self.all_rewards[self.a_t][self.M:]))
        gp, gm = 0, 0
        for sp_, sm_ in zip(sp, sm):
            gp, gm = max([0, gp + sp_]), max([0, gm + sm_])
            if max([gp, gm]) >= self.h:
                self.count+=1
                print(self.count, "al t:",self.t)
                return True
        return False

    def reset(self):
        self.__init__(*self._original_params)



#------------------------------TS-----------------------------------------------

class TSAgent():
    def __init__(self, K, ncostumers):
        self.K = K
        self.a_t = None
        self.alpha, self.beta = np.ones(K), np.ones(K)
        self.N_pulls = np.zeros(K)
        self._original_params =(K,ncostumers)
        self.n_costumers = ncostumers

    def pull_arm(self):
        theta = np.random.beta(self.alpha, self.beta)
        self.a_t = np.argmax(theta)
        return self.a_t

    def update(self, r_t):
        r_t=r_t/self.n_costumers
        self.alpha[self.a_t] += r_t
        self.beta[self.a_t] += 1 - r_t
        self.N_pulls[self.a_t] += 1

    def reset(self):
        self.__init__(*self._original_params)

#--------------------TS-SW----------------------
class TSSWAgent():
    def __init__(self, K,W,ncostumers):
        self.K = K
        self.a_t = None
        self.alpha, self.beta = np.ones(K), np.ones(K)
        self.N_pulls = np.zeros(K)
        self.n_costumers = ncostumers
        self._original_params =(K,W,ncostumers)
        self.t=0
        self.W=W
        self.reward_queue = []
        self.refArm_queue = []

    def pull_arm(self):
        theta = np.random.beta(self.alpha, self.beta)
        self.a_t = np.argmax(theta)
        return self.a_t

    def update(self, r_t):
        r_t=r_t/self.n_costumers
        self.alpha[self.a_t] += r_t
        self.beta[self.a_t] += 1 - r_t
        self.N_pulls[self.a_t] += 1


        self.reward_queue.append(r_t)
        self.refArm_queue.append(self.a_t)
        if (self.t > self.W):
            queueReward = self.reward_queue.pop(0)
            armtoModify = self.refArm_queue.pop(0)
            self.alpha[armtoModify] -= queueReward
            self.beta[armtoModify] -= (1 - queueReward)

    def reset(self):
        self.__init__(*self._original_params)

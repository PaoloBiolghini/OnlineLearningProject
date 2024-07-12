import numpy as np
from scipy import optimize

class UCBAgent:
    def __init__(self,valuation, budget,n_users, T , K, ):
        self.valuation = valuation
        self.budget = budget
        self.T = T 

        self.t = 0
        self.n_users = n_users
        self.rho = self.budget/(self.T*self.n_users)
        self.K=K
        self.N_pulls = np.zeros(self.K)
        self.bids = np.linspace(0, 1, K)
        self.f_t = np.zeros(self.K)
        self.c_t = np.zeros(self.K)
        self.f_t_UCB = np.zeros(self.K)
        self.c_t_LCB = np.zeros(self.K)
        self.arm=None
        self.arm_history=[]
        self.gamma_history=[]
    
    def __clacoulate_bounds(self):
        ft_mean = np.zeros(self.K)
        ct_mean = np.zeros(self.K)  
        for i in range(self.K):
            if self.N_pulls[i] == 0:
                ft_mean[i] = 0
                ct_mean[i] = 0
                self.f_t_UCB[i] = 100000
                self.c_t_LCB[i] = -100000
            else:
                ft_mean[i] = self.f_t[i]/self.N_pulls[i]
                ct_mean[i] = self.c_t[i]/self.N_pulls[i]
                self.f_t_UCB[i] = ft_mean[i] + np.sqrt(2*np.log(self.T)/self.N_pulls[i])
                self.c_t_LCB [i]= ct_mean[i] - np.sqrt(2*np.log(self.T)/self.N_pulls[i])
        
        return self.f_t_UCB, self.c_t_LCB
    def __solve_problem(self):
        c = -(self.f_t_UCB)
        A_ub = [self.c_t_LCB]
        b_ub = [self.rho]
        A_eq = [np.ones(len(self.c_t_LCB))]
        b_eq = [1]
        res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0,1))
        gamma = res.x
        self.gamma_history.append(gamma)
        return gamma
    def __sample(self, gamma):
        self.arm = np.random.choice(np.arange(self.K), p=gamma)
        self.N_pulls[self.arm] += 1
        # if self.budget >= 1:
        #     return self.arm
        # else: return 0
        return self.arm

    
    def bid(self):
        if self.budget < 1:
            self.arm_history.append(0)
            return 0
            
        else:
            self.__clacoulate_bounds()
            gamma = self.__solve_problem()
            arm=self.__sample(gamma)
            self.arm_history.append(arm)
            return self.bids[arm]
    
    def update(self, f_t, c_t):
        self.f_t[self.arm] += f_t
        self.c_t[self.arm] += c_t
        self.budget -= c_t
        
    def update_per_round(self):
        self.t += 1
        if self.t < self.T:
            self.rho = self.budget/((self.T-self.t)*self.n_users)
        else:
            self.rho = self.budget/(self.n_users)


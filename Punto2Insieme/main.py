#%%
import numpy as np
import matplotlib.pyplot as plt

from AdversarialPricingEnviroment import NonstationaryAdvEnvironment 
from AdversarialPricingEnviroment import PricingAdversarialEnvironment as PricingEnvironment
from agent import EXP3Agent,EXP3PAgent, EXP3AgentSlidingWindow as EXP3SW
import math
from visualization import showPlotRegrets,showCombinedPlots,showPlotPulls
from scipy import stats
from scipy import optimize

from FirstPriceAuction import FirstPriceAuction
from FFMultiplicativePacingAgent import FFMultiplicativePacingAgent


def bids_generator(n, T, plot=True, step=20):
    other_bids = np.zeros((n, T))

    for idx in range(0, T, step):
        avg = np.random.uniform(0.2, 0.8, 1)

        end_idx = min(idx + step, T)
        other_bids[:, idx:end_idx] = np.random.uniform(avg - 0.2, avg + 0.2, (n, end_idx - idx))

    if plot:
        plt.figure(figsize=(10, 6))
        for i in range(n):
            plt.plot(range(T), other_bids[i, :], label=f'Advertiser {i}')
        plt.xlabel('Time')
        plt.ylabel('Bids')
        plt.legend()
        plt.show()

    return other_bids


def clairvoyant(n, T, val, rho, K, other_bids, step=20):
    expected_clairvoyant_utilities = []
    expected_clairvoyant_bids = []

    for idx in range(0, T, step):
        available_bids = np.linspace(0, 1, K)

        end_idx = min(idx + step, T)
        win_probabilities = np.zeros(K)
        for bid_idx in range(K):
            bid = available_bids[bid_idx]
            win_probabilities[bid_idx] = np.mean(bid > other_bids[:, idx:end_idx])

        c = -(val - available_bids) * win_probabilities
        A_ub = [available_bids * win_probabilities]
        b_ub = [rho]
        A_eq = [np.ones(len(available_bids))]
        b_eq = [1]

        res = optimize.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1))
        gamma = res.x

        expected_clairvoyant_utilities += [-res.fun] * (end_idx - idx)
        expected_clairvoyant_bids += [sum(available_bids * gamma * win_probabilities)] * (end_idx - idx)

    return expected_clairvoyant_bids, expected_clairvoyant_utilities

def runComputation(agent, env, n_trials):
    regret_per_trial = []

    n_advertisers = 4
    ctrs = np.ones(n_advertisers)
    my_valuation = 0.6
    B = 1000

    # environmental settings
    days=300
    auctions_per_day = 60


    eta = 1 / np.sqrt(T*auctions_per_day)
    rho = B / (T*auctions_per_day)

    for seed in range(n_trials):
        np.random.seed(seed)

        agentbids = FFMultiplicativePacingAgent(K=K,
                                            valuation=my_valuation,
                                            budget=B,
                                            T=T,
                                            eta=eta)

        auction = FirstPriceAuction(ctrs)
        winhistory=[]
        utilities = np.array([])
        my_bids = np.array([])
        my_payments = np.array([])

        other_bids = bids_generator(n_advertisers - 1, T, step=100,plot=False)
        expected_clairvoyant_bids, expected_clairvoyant_utilities = clairvoyant(n_advertisers, T, my_valuation, rho, K,
                                                                                other_bids, step=100)

        env.reset()
        agent.reset()

        agent_rewards = np.array([])

        all_regrets = []
        all_payments = []

        m_t = other_bids.max(axis=0)

        for t in range(T):
            costumersOfDay=0
            for i in range(auctions_per_day):
                my_bid = agentbids.bid()
                bids = np.append(my_bid, other_bids[:, t].ravel())
                winners, payments_per_click = auction.round(bids=bids)
                my_win = int(winners == 0)
                if my_win==1:
                    costumersOfDay+=1
                f_t, c_t = (my_valuation - m_t[t]) * my_win, m_t[t] * my_win
                agentbids.update(f_t, c_t, m_t[t])


            a_t = agent.pull_arm()
            scelta = prices[a_t]

            if(costumersOfDay>0):
                d_t, r_t = env.round(scelta, costumersOfDay)
                agent.update(r_t,costumersOfDay)
                # print("entra",costumersOfDay, "  r:",r_t)
                agent_rewards = np.append(agent_rewards, r_t)
            else:
                agent.skip()
                agent_rewards = np.append(agent_rewards, 0)
            winhistory.append(costumersOfDay)
        realExpected=np.array(expected_clairvoyant_rewards)*np.array(winhistory)
        print("AUCTION WINNED:",sum(np.array(winhistory)))
        cumulative_regret = np.cumsum(agent_rewards)
        regret_per_trial.append(cumulative_regret)

    return regret_per_trial
    
#------------General-Parameters------------------
T = 300
K = 50
n_trials = 5
n_customers = 10 # I assume the number of customers arriving is the same everyday (for now, in general this is not true)
prices = np.linspace(0,1,K)

#-----------------env setting----------------------
cost = 0.1
regret_per_trial = []
        
opt_lRate=math.sqrt(np.log(K)/(K*T))

conversion_probability = lambda p,t: (1-p**(1/5+5*t/T))/(1+p)
reward_function = lambda price, n_sales: (price-cost)*n_sales
maximum_profit = reward_function(max(prices), n_customers)
profit_curve = lambda p,t:(1-p**(1/5+5*t/T))/(1+p)*(p-cost)

#------------compute clairvoyant---------------
sum_expcted_rewards = np.zeros(K)
for n in range(T):
    sum_expcted_rewards += profit_curve(prices, n)

best_price_index = np.argmax(sum_expcted_rewards)
best_price = prices[best_price_index]
print("BEST PRICE:",best_price_index, "VALUE:",best_price)
expected_clairvoyant_rewards=[]
for n in range(T):
    expected_clairvoyant_rewards.append(profit_curve(best_price,n)*n_customers)
#-----------------------------------------------
env = PricingEnvironment(conversion_probability=conversion_probability, cost=cost)
#env = NonstationaryAdvEnvironment(self, T, n_users, seed, n_arms)

#%%--------EXP3----------------------------
exp3_agent = EXP3Agent(K, opt_lRate)
regret_per_trial=runComputation(exp3_agent,env,n_trials)

x=np.linspace(0,T,100)
y= np.sqrt(x)*math.sqrt(K*math.log10(K))

showPlotRegrets(regret_per_trial,"EXP3 Regret",T,n_trials,None)
showPlotPulls(exp3_agent,"EXP3 Agent",K,best_price_index)

#%%--------EXP3 sliding window-------------
# W=300 #optimal window size
#
# exp3_agentsw = EXP3SW( K, opt_lRate, W, n_customers)
# regret_per_trialsw=runComputation(exp3_agentsw,env,n_trials)
#
#
# showPlotRegrets(regret_per_trialsw,"EXP3 Sliding Window Regret",T,n_trials)
# showPlotPulls(exp3_agentsw,f"EXP3 SW{int(W)} Agent",K, best_price_index)
#
# showCombinedPlots(regret_per_trial,exp3_agent,best_price_index,"EXP3",regret_per_trialsw,exp3_agentsw,best_price_index,f"EXP3 SW{W}",T,n_trials)
# #%%-----------EXP3.P-----------------------
delta=0.8
gamma = 1.05*math.sqrt(K*math.log(K)/T)  # Exploration rate
beta = math.sqrt(math.log(K*delta)/(K*T))  # Perturbation factor
eta = 0.95*math.sqrt(math.log(K)/(K*T))    # Learning rate

gamma=0.1
beta=0.01
eta=0.1

exp3p = EXP3PAgent(K, gamma, beta, eta, n_customers)
regret_per_trialp=runComputation(exp3p,env,n_trials)

showPlotRegrets(regret_per_trialp,"EXP3.P Regret",T,n_trials)
showPlotPulls(exp3p,"EXP3.P Agent",K, best_price_index)

showCombinedPlots(regret_per_trial,exp3_agent,best_price_index,"EXP3 Reward Over Time",regret_per_trialp,exp3p,best_price_index,"EXP3.P Reward Over Time",T,n_trials)
#showCombinedPlots(regret_per_trial,exp3_agentsw,best_price_index,f"EXP3 SW{W}",regret_per_trialp,exp3P_agent,best_price_index,"EXP3.P",T,n_trials)





'''
T = 1000 # try T=100, why this behavior?
K = 10
n_trials = 5
prices = np.linspace(0,1,K) # 100 actions!
cost = 0.1
conversion_probability = lambda p,t: (1-p**(1/5+5*t/T))/(1+p)

n_customers = 100 # I assume the number of customers arriving is the same everyday (for now, in general this is not true)

reward_function = lambda price, n_sales: (price-cost)*n_sales
maximum_profit = reward_function(max(prices), n_customers) # the maximum possible reward is selling at the maximum price to every possible customer

# let's compute the clairvoyant
profit_curve = lambda p,t:(1-p**(1/5+5*t/T))/(1+p)*(p-cost)

sum_expcted_rewards = np.zeros(K)
for n in range(T):
    sum_expcted_rewards += profit_curve(prices, n)


best_price_index = np.argmax(sum_expcted_rewards)
print("BEST PRICE:",best_price_index)
best_price = prices[best_price_index]
print("BEST PRICE:",best_price_index, "VALUE:",best_price)
expected_clairvoyant_rewards=[]
for n in range(T):
    expected_clairvoyant_rewards.append(profit_curve(best_price,n)*n_customers)


regret_per_trial = []

opt_lRate=math.sqrt(np.log(K)/(K*T))
opt_window=math.sqrt(T)

for seed in range(n_trials):
    np.random.seed(seed)
    env = PricingEnvironment(conversion_probability=conversion_probability, cost=cost)
    
    agent = EXP3Agent(K, opt_lRate)
    #agent = EXP3SW(K,opt_lRate,17)
    
    gamma = 0.1  # Exploration rate
    beta = 0.01  # Perturbation factor
    eta = 0.1    # Learning rate

    #agent = EXP3PAgent(K, gamma, beta, eta)

    agent_rewards = np.array([])

    for t in range(T):
        pi_t = agent.pull_arm() ## the agent returns the index!!
        p_t = prices[pi_t] # I get the actual price
        
        d_t, r_t = env.round(p_t, n_customers)
        
        agent.update(1-r_t)

        agent_rewards = np.append(agent_rewards, r_t)
    
    
    cumulative_regret = np.cumsum(expected_clairvoyant_rewards-agent_rewards)
    regret_per_trial.append(cumulative_regret)

regret_per_trial = np.array(regret_per_trial)

average_regret = regret_per_trial.mean(axis=0)
regret_sd = regret_per_trial.std(axis=0)

plt.plot(np.arange(T), average_regret, label='Average Regret')
plt.title('cumulative regret of EXP3 sliding window')
plt.fill_between(np.arange(T),
                average_regret-regret_sd/np.sqrt(n_trials),
                average_regret+regret_sd/np.sqrt(n_trials),
                alpha=0.3,
                label='Uncertainty')
#plt.plot((0,T-1), (average_regret[0], average_regret[-1]), 'ro', linestyle="--")
plt.xlabel('$t$')
plt.legend()
plt.show()

plt.figure()
plt.barh(np.arange(100), agent.N_pulls)
plt.axhline(best_price_index, color='red', label='Best price')
plt.ylabel('actions')
plt.xlabel('numer of pulls')
plt.legend()
plt.title('Number of pulls for each action')
plt.show()'''


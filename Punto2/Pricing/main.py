#%%
import numpy as np
import matplotlib.pyplot as plt

from AdversarialPricingEnviroment import NonstationaryAdvEnvironment 
from AdversarialPricingEnviroment import PricingAdversarialEnvironment as PricingEnvironment
from agent import EXP3Agent,EXP3PAgent, EXP3AgentSlidingWindow as EXP3SW
import math
from visualization import showPlotRegrets,showCombinedPlots,showPlotPulls
    
def runComputation(agent, env, n_trials):
    regret_per_trial = []

    for seed in range(n_trials):
        np.random.seed(seed)
        env.reset()
        agent.reset()

        agent_loss = np.array([])

        for t in range(T):
            a_t = agent.pull_arm()
            p_t = prices[a_t]
            d_t, r_t = env.round(p_t, n_customers)
            agent.update(r_t)

            agent_loss = np.append(agent_loss, (1-r_t))

        cumulative_regret = np.cumsum(expected_clairvoyant_rewards - agent_loss)
        regret_per_trial.append(cumulative_regret)

    return regret_per_trial
    
#------------General-Parameters------------------
T = 1000 
K = 10
n_trials = 5
n_customers = 100 # I assume the number of customers arriving is the same everyday (for now, in general this is not true)
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

showPlotRegrets(regret_per_trial,"EXP3 Regret",T,n_trials)
showPlotPulls(exp3_agent,"EXP3 Agent",K,best_price_index)

#%%--------EXP3 sliding window-------------
W=math.sqrt(T) #optimal window size

exp3_agentsw = EXP3SW( K, opt_lRate, W)
regret_per_trialsw=runComputation(exp3_agentsw,env,n_trials)


showPlotRegrets(regret_per_trialsw,"EXP3 Sliding Window Regret",T,n_trials)
showPlotPulls(exp3_agentsw,f"EXP3 SW{W} Agent",K, best_price_index)

showCombinedPlots(regret_per_trial,exp3_agent,best_price_index,"EXP3",regret_per_trialsw,exp3_agentsw,best_price_index,f"EXP3 SW{W}",T,n_trials)
#%%-----------EXP3.P-----------------------
gamma = 0.1  # Exploration rate
beta = 0.01  # Perturbation factor
eta = 0.1    # Learning rate

exp3P_agent = EXP3PAgent(K, gamma, beta, eta)
regret_per_trialp=runComputation(exp3P_agent,env,n_trials)

showPlotRegrets(regret_per_trialsw,"EXP3.P Regret",T,n_trials)
showPlotPulls(exp3P_agent,"EXP3.P Agent",K, best_price_index)

showCombinedPlots(regret_per_trial,exp3_agent,best_price_index,"EXP3",regret_per_trialp,exp3P_agent,best_price_index,"EXP3.P",T,n_trials)
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


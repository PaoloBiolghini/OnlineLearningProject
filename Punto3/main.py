import numpy as np
from env import PricingSlightlyNonstationaryEnvironment as PricingEnv
from agents import UCB1Agent,SWUCBAgent
import matplotlib.pyplot as plt


def showPlotRegrets(regret_per_trial,title):
    regret_per_trial = np.array(regret_per_trial)

    average_regret = regret_per_trial.mean(axis=0)
    regret_sd = regret_per_trial.std(axis=0)

    plt.plot(np.arange(T), average_regret, label='Average Regret')
    plt.title(title)
    plt.fill_between(np.arange(T),
                     average_regret - regret_sd / np.sqrt(n_trials),
                     average_regret + regret_sd / np.sqrt(n_trials),
                     alpha=0.3,
                     label='Uncertainty')
    # plt.plot((0,T-1), (average_regret[0], average_regret[-1]), 'ro', linestyle="--")
    plt.xlabel('$t$')
    plt.legend()
    plt.show()

def showPlotPulls(agent,title):
    plt.figure()
    plt.barh(np.arange(100), agent.N_pulls)
    plt.axhline(best_price_index, color='red', label='Best price')
    plt.ylabel('actions')
    plt.xlabel('numer of pulls')
    plt.legend()
    plt.title('Number of pulls for each action '+title)
    plt.show()



#------------General-Paramiters------------------
T = 10000
K = 100
n_customers = 100
n_trials = 20

prices = np.linspace(0, 1, K)


#-----------------env setting---------------
cost = 0.1
variance=0.1

conversion_probability = lambda p,t: (1-p**(3/5+4/5*t/T))/(1+p)
profit_curve = lambda p,t: (1-p**(3/5+4/5*t/T))/(1+p) * (p - cost)

#------------compute clairvoyant---------------
sum_expcted_rewards = np.zeros(K)
for n in range(T):
    sum_expcted_rewards += profit_curve(prices, n)

best_price_index = np.argmax(sum_expcted_rewards)
print("BEST PRICE:", best_price_index)
best_price = prices[best_price_index]
print("BEST PRICE:", best_price_index, "VALUE:", best_price)
expected_clairvoyant_rewards = []
for n in range(T):
    expected_clairvoyant_rewards.append(profit_curve(best_price, n) * n_customers)
#----------------------------------------------


#-----------------Computation-------------
regret_per_trial = []

ucb_agent=None

for seed in range(n_trials):
    np.random.seed(seed)
    env = PricingEnv(conversion_probability, cost, variance)
    ucb_agent = UCB1Agent(K, T)

    agent_rewards = np.array([])

    for t in range(T):
        a_t = ucb_agent.pull_arm()
        d_t,r_t = env.round(a_t,n_customers)
        ucb_agent.update(r_t)

        agent_rewards = np.append(agent_rewards, r_t)

    cumulative_regret = np.cumsum(expected_clairvoyant_rewards-agent_rewards)
    regret_per_trial.append(cumulative_regret)

#----------------Final-Plots---------------------

showPlotRegrets(regret_per_trial,"UCB1 Regret")
showPlotPulls(ucb_agent,"UCB1 Agent")



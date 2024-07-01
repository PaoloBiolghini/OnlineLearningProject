import math

import numpy as np
from env import PricingSlightlyNonstationaryEnvironment as PricingEnv
from agents import UCB1Agent,SWUCBAgent,CUSUMUCBAgent,TSAgent,TSSWAgent,SWUCBMixedAgent
import matplotlib.pyplot as plt
from visualization import showPlotRegrets,showCombinedPlots,showPlotPulls

def runComputation(agent, env, n_trials):
    regret_per_trial = []

    for seed in range(n_trials):
        np.random.seed(seed)
        env.reset()
        agent.reset()

        agent_rewards = np.array([])

        for t in range(T):
            a_t = agent.pull_arm()
            p_t = prices[a_t]
            d_t, r_t = env.round(p_t, n_customers)
            agent.update(r_t)

            agent_rewards = np.append(agent_rewards, r_t)

        cumulative_regret = np.cumsum(expected_clairvoyant_rewards - agent_rewards)
        regret_per_trial.append(cumulative_regret)

    return regret_per_trial


#------------General-Parameters------------------
T = 100000
K = 100
n_customers = 100
n_trials = 3

prices = np.linspace(0, 1, K)

#-----------------env setting----------------------
cost = 0.1
variance: float=0.1
nstep=10

conversion_probability = lambda p,t: (1-p**(2+2*(t//nstep)*nstep/T))
profit_curve = lambda p,t: (1-p**(2+2*(t//nstep)*nstep/T)) * (p - cost)

#------------compute clairvoyant---------------
sum_expcted_rewards = np.zeros(K)
for n in range(T):
    sum_expcted_rewards += profit_curve(prices, n)

best_price_index = np.argmax(sum_expcted_rewards)
best_price = prices[best_price_index]
print("BEST PRICE:", best_price_index, "VALUE:", best_price)

expected_clairvoyant_rewards = []
for n in range(T):
    expected_clairvoyant_rewards.append(profit_curve(best_price, n) * n_customers)
#----------------------------------------------

env = PricingEnv(conversion_probability, cost, variance)

#----------------UCB-Computation------------------------------------------------------

ucb_agent = UCB1Agent(K, T)

regret_per_trial=runComputation(ucb_agent,env,n_trials)

showPlotRegrets(regret_per_trial,"UCB1 Regret",T,n_trials)
showPlotPulls(ucb_agent,"UCB1 Agent",K,best_price_index)

#-----------UCB-SW-Computation-----------------------------------------------------------
W=500

ucb_agentsw = SWUCBAgent(K, T, W)
regret_per_trialsw=runComputation(ucb_agentsw,env,n_trials)



showPlotRegrets(regret_per_trialsw,"UCB1 Sladiding Window Regret",T,n_trials)
showPlotPulls(ucb_agentsw,"UCB1 SW500 Agent",K, best_price_index)

showCombinedPlots(regret_per_trial,ucb_agent,best_price_index,"UCB1",regret_per_trialsw,ucb_agentsw,best_price_index,"UCB1 SW500",T,n_trials)

#------------UBC-SW-MIXED---------

ucb_agentswmix = SWUCBMixedAgent(K, T, W)
regret_per_trialswmix=runComputation(ucb_agentswmix,env,n_trials)



showPlotRegrets(regret_per_trialswmix,"UCB1 Sladiding Window MIXED Regret",T,n_trials)
showPlotPulls(ucb_agentswmix,"UCB1 SW500 Agent",K, best_price_index)

showCombinedPlots(regret_per_trialswmix,ucb_agentswmix,best_price_index,"UCB1 MIXED",regret_per_trialsw,ucb_agentsw,best_price_index,"UCB1 SW500",T,n_trials)



#-----------------CUM-SUM-UCB---------------
# U_T = 0.1 # maximum number of abrupt changes
# h = 2*np.log(T/U_T) # sensitivity of detection, threshold for cumulative deviation
# alpha = np.sqrt(U_T*np.log(T/U_T)/T) # probability of extra exploration
#
# M = int(np.log(T/U_T)) # robustness of change detection
#
# #-----------------Computation-------------
# ucb_cumsum = CUSUMUCBAgent(K, T, M, h, alpha)
# regret_per_trial_cumsum=runComputation(ucb_cumsum,env,n_trials)
#
# showPlotRegrets(regret_per_trial_cumsum,"UCB1 CUM SUM 3 REGRET",T,n_trials)
# showPlotPulls(ucb_cumsum,"UCB1 CUM SUM 3",K,best_price_index)

#----------------------------TS-------------------------
# tsAgent=TSAgent(K,n_customers)
# regretTS=runComputation(tsAgent,env,n_trials)
#
# showPlotRegrets(regretTS,"TS REGRET",T,n_trials)
# showPlotPulls(tsAgent,"TS 3",K,best_price_index)
#
#
# #----------------------------TS--SW-----------------------
# W=1000
# tsAgentSW=TSSWAgent(K,W,n_customers)
# regretTSSW=runComputation(tsAgent,env,n_trials)
#
# showPlotRegrets(regretTSSW,"TS SW REGRET",T,n_trials)
# showPlotPulls(tsAgentSW,"TS SW",K,best_price_index)





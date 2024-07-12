import math

import numpy as np
import matplotlib.pyplot as plt

def showPlotRegrets(regret_per_trial,title,T,n_trials, baseline=None):
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
    if(baseline != None):
        plt.plot(baseline[0], baseline[1], label='sqrt(T)', color='blue')

    # plt.plot((0,T-1), (average_regret[0], average_regret[-1]), 'ro', linestyle="--")
    plt.xlabel('$t$')
    plt.legend()
    plt.show()

def showPlotPulls(agent,title,K,best_price_index):
    plt.figure()
    plt.barh(np.arange(K), agent.N_pulls)
    plt.axhline(best_price_index, color='red', label='Best price')
    plt.ylabel('actions')
    plt.xlabel('numer of pulls')
    plt.legend()
    plt.title('Number of pulls for each action '+title)
    plt.show()


def showCombinedPlots(regret_per_trial_1, agent_1, best_price_index_1, title_1,
                      regret_per_trial_2, agent_2, best_price_index_2, title_2,T,n_trials,baseline=None):
    regret_per_trial_1 = np.array(regret_per_trial_1)
    regret_per_trial_2 = np.array(regret_per_trial_2)

    average_regret_1 = regret_per_trial_1.mean(axis=0)
    regret_sd_1 = regret_per_trial_1.std(axis=0)

    average_regret_2 = regret_per_trial_2.mean(axis=0)
    regret_sd_2 = regret_per_trial_2.std(axis=0)

    # Create a 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # First plot: Regret for agent_1
    axs[0, 0].plot(np.arange(T), average_regret_1, label='Average Regret')
    if(baseline!=None):
        axs[0, 0].plot(baseline[0], baseline[1], label='UpperBound', color='blue')
        axs[1, 0].plot(baseline[0], baseline[1], label='UpperBound', color='blue')
    axs[0, 0].fill_between(np.arange(T),
                           average_regret_1 - regret_sd_1 / np.sqrt(n_trials),
                           average_regret_1 + regret_sd_1 / np.sqrt(n_trials),
                           alpha=0.3,
                           label='Uncertainty')
    axs[0, 0].set_title(title_1 + " Regret")
    axs[0, 0].set_xlabel('$t$')
    axs[0, 0].legend()

    # Second plot: Pulls for agent_1
    axs[0, 1].barh(np.arange(len(agent_1.N_pulls)), agent_1.N_pulls[::-1])
    axs[0, 1].axhline(best_price_index_1, color='red', label='Best price')
    axs[0, 1].set_ylabel('Actions')
    axs[0, 1].set_xlabel('Number of pulls')
    axs[0, 1].legend()
    axs[0, 1].set_title("Number of pulls for each action - " + title_1)

    # Third plot: Regret for agent_2
    axs[1, 0].plot(np.arange(T), average_regret_2, label='Average Regret')
    axs[1, 0].fill_between(np.arange(T),
                           average_regret_2 - regret_sd_2 / np.sqrt(n_trials),
                           average_regret_2 + regret_sd_2 / np.sqrt(n_trials),
                           alpha=0.3,
                           label='Uncertainty')
    axs[1, 0].set_title(title_2 + " Regret")
    axs[1, 0].set_xlabel('$t$')
    axs[1, 0].legend()

    # Fourth plot: Pulls for agent_2
    axs[1, 1].barh(np.arange(len(agent_2.N_pulls)), agent_2.N_pulls)
    axs[1, 1].axhline(best_price_index_2, color='red', label='Best price')
    axs[1, 1].set_ylabel('Actions')
    axs[1, 1].set_xlabel('Number of pulls')
    axs[1, 1].legend()
    axs[1, 1].set_title("Number of pulls for each action - " + title_2)

    plt.tight_layout()
    plt.show()
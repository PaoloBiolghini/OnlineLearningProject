#%%
import numpy as np
import matplotlib.pyplot as plt

from AdversarialPricingEnviroment import NonstationaryAdvEnvironment 
from AdversarialPricingEnviroment import PricingAdversarialEnvironment as PricingEnvironment
from UCB1Agent import UCB1Agent
class Main():
    def test_invironment():
        T = 5
        n_users = 2
        seed = 1
        n_arms = 3
        env = NonstationaryAdvEnvironment(T, n_users, seed, n_arms)

        i = 0
        while i < T:
            # Assess price (ex: p = 0.5 fixed)
            arm = 2
            print(f'Day {i+1} started. Price = {0.5}.')
            prob = env.get_prob(arm)
            print(f'Current Purchase Probability for every user: {prob}.\n')
            
            i += 1
    def test_UCB1Agent():
        T = 10000 # try T=100, why this behavior?
        K = 100

        prices = np.linspace(10,20,K) # 100 actions!
        cost = 10
        conversion_probability = lambda p: 1-p/20

        n_customers = 100 # I assume the number of customers arriving is the same everyday (for now, in general this is not true)

        reward_function = lambda price, n_sales: (price-cost)*n_sales
        maximum_profit = reward_function(max(prices), n_customers) # the maximum possible reward is selling at the maximum price to every possible customer

        # let's compute the clairvoyant
        profit_curve = reward_function(prices, n_customers*conversion_probability(prices))
        best_price_index = np.argmax(profit_curve)
        best_price = prices[best_price_index]
        expected_clairvoyant_rewards = np.repeat(profit_curve[best_price_index], T)

        n_trials = 100

        regret_per_trial = []

        for seed in range(n_trials):
            np.random.seed(seed)
            env = PricingEnvironment(conversion_probability=conversion_probability, cost=cost)
            ucb_agent = UCB1Agent(K, T, range=maximum_profit)

            agent_rewards = np.array([])

            for t in range(T):
                pi_t = ucb_agent.pull_arm() ## the agent returns the index!!
                p_t = prices[pi_t] # I get the actual price
                d_t, r_t = env.round(p_t, n_customers)
                ucb_agent.update(r_t)

                agent_rewards = np.append(agent_rewards, r_t)

            cumulative_regret = np.cumsum(expected_clairvoyant_rewards-agent_rewards)
            regret_per_trial.append(cumulative_regret)

        regret_per_trial = np.array(regret_per_trial)

        average_regret = regret_per_trial.mean(axis=0)
        regret_sd = regret_per_trial.std(axis=0)

        plt.plot(np.arange(T), average_regret, label='Average Regret')
        plt.title('cumulative regret of UCB1')
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
        plt.barh(np.arange(100), ucb_agent.N_pulls)
        plt.axhline(best_price_index, color='red', label='Best price')
        plt.ylabel('actions')
        plt.xlabel('numer of pulls')
        plt.legend()
        plt.title('Number of pulls for each action')
        plt.show()



if __name__ == "__main__":
    #Main.test_invironment()
    Main.test_UCB1Agent()
# %%

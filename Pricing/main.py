from GPThompson import GPThompson
from GPUCBAgent import GPUCBAgent
from StochasticPricingEnvironment import StochasticPricingEnvironment
import numpy as np 

def discretize(T):
    epsilon = T**(-0.33)
    K = int(1/epsilon)
    return K

def rescale(x, min_x, max_x):
    return min_x + (max_x-min_x)*x


if __name__ == '__main__':
    
    min_price, max_price = 10, 20
    n_customers = 100
    cost = 10
    
    T= 200
    
    K = discretize(T)
    prices = np.linspace(min_price, max_price, K)
    
    
    # reward function
    reward_function = lambda price, n_sales: (price-cost)*n_sales
    maximum_profit = reward_function(max(prices), n_customers) # the maximum possible reward is selling at the maximum price to every possible customer

    # conversion prob
    conversion_probability = lambda p: 1-p/20 #TODO: try to change it
    
    # clairvoyant
    profit_curve = reward_function(prices, n_customers*conversion_probability(prices))
    best_price_index = np.argmax(profit_curve)
    best_price = prices[best_price_index]
    expected_clairvoyant_rewards = np.repeat(profit_curve[best_price_index], T)
    
    # initialize agennt and environment
    gp_agent = GPUCBAgent(T, discretization=K)
    env = StochasticPricingEnvironment(conversion_probability=conversion_probability, cost=cost)
    gp_agent_rewards = np.array([])
    for t in range(T):
        p_t = gp_agent.pull_arm()
        p_t = rescale(p_t, min_price, max_price)
        d_t, r_t = env.round(p_t, n_t=n_customers)
        gp_agent.update(r_t/n_customers)
        gp_agent_rewards = np.append(gp_agent_rewards, r_t)
        print(f"day: {t}")
        print(f"price: {p_t}")
        print(f"revenue: {r_t}")
        print("----------------------------")
    print(f"prices: {prices}")
        

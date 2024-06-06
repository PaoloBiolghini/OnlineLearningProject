#%%
from AdversarialPricingEnviroment import PricingAdversarialEnvironment as PricingEnvironment
import numpy as np
import matplotlib.pyplot as plt
import scipy.special

#settings
T=300
cost = 0.1
n_customers=100
prices = np.linspace(0, 1, 100)
r=5


conversion_probability = lambda p,t:1- np.exp(3*p*127/T)/np.exp(3)
conversion_probability = lambda p,t: 1-scipy.special.expit(5*p*200/T)
#Proposta da me
conversion_probability = lambda p,t: (1-p**(1/5+5*t/T))/(1+p)

#creating the env
env = PricingEnvironment(conversion_probability=conversion_probability, cost=cost)

#computing the expected profit curve
expected_profit_curve = n_customers*conversion_probability(prices,1)*(prices-cost)
#computing the estimated profit curve using the enviroment
demand,estimated_profit_curve=env.round(prices, n_customers)

#finding best arm
best_price_index = np.argmax(expected_profit_curve)
best_price = prices[best_price_index]

#plotting
plt.figure()
plt.plot(prices, expected_profit_curve, label='Expected Profit Curve')
plt.plot(prices, estimated_profit_curve, label='Estimated Profit Curve')
plt.scatter(best_price, expected_profit_curve[best_price_index], color='red', s=50)
plt.xlabel('Item Price')
plt.title('Expected and Estimated Profit Curves time=1')
plt.legend()
plt.show()
# %%

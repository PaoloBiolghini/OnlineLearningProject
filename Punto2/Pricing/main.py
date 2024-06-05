#%%
from AdversarialPricingEnviroment import NonstationaryAdvEnvironment 
class Main():
    def __init__(self) -> None:
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
if __name__ == "__main__":
    Main()
# %%

#%%
from AdversarialAuctionEnviroment import NonstationaryAdvEnvironment 
class Main():
    def __init__(self) -> None:
        T = 5
        n_competitors = 2
        auctions_per_day = 2
        seed = 1

        env = NonstationaryAdvEnvironment(T, n_competitors, auctions_per_day, seed)

        i = 0
        while i < T:
            # Assess price (ex: p = 0.5 fixed)
            arm = 50
            
            for j in range(i*auctions_per_day, (i+1)*auctions_per_day):
                bid = env.get_bid()
                print(f'Auction {j+1} started. Competitors Bids: {bid}.')
                
                # Assess auction (ex: 1 slot won)
                slot_won = 1
                print(f'Number of AdvSlots won: {slot_won}.')
                
                env.next_round(slot_won, arm)
                print(f'Round Reward: {env.current_reward}.\n')
            
            print(f'Cumulative Reward at the end of day {i}: {env.cumulative_reward}.\n')
            print('------------------------------------------------------------------------')
            
            i += 1
if __name__ == "__main__":
    Main()
# %%

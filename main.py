# import Agent, Auction
from Auction.Agent import Agent
from Auction.RandomAgent import RandomAgent
from Auction.SecondPriceAuction import SecondPriceAuction
import numpy as np


def loop_auction_day(auction, agent, other_bids, n_users=1000):
    utilities = np.array([])
    my_bids = np.array([])
    my_payments = np.array([])
    total_wins = 0
    m_t = other_bids.max(axis=0)
    
    np.random.seed(18)
    for u in range(n_users):
        # interaction
        my_bid = agent.bid()
        bids = np.append(my_bid, other_bids[:, u].ravel())
        winners, payments_per_click = auction.round(bids=bids)
        my_win = int(winners==0)
        f_t, c_t = (my_valuation-m_t[u])*my_win, m_t[u]*my_win
        agent.update(f_t, c_t)
        # logging
        utilities = np.append(utilities, f_t)
        my_bids = np.append(my_bids, my_bid)
        my_payments = np.append(my_payments, c_t)
        total_wins+=my_win
    print(f'Total # of Wins: {total_wins}')
    return utilities, my_bids, my_payments, total_wins

if __name__ == '__main__':
    # TODO: set values with parameters
    # auction parameters
    n_advertisers = 4 
    ctrs = np.ones(n_advertisers)
    my_valuation = 0.8
    B = 150

    # environmental settings
    n_users = 1000

    # competitors in stochastic environment
    other_bids = np.random.uniform(0, 1, size = (n_advertisers-1, n_users))
    # noise in the environment
    eta = 1/np.sqrt(n_users)
    
    # auction agent and auction
    auction_agent = Agent(
        valuation=my_valuation,
        budget=B,
        T=n_users, 
        eta=eta
    )
    
    # random agent
    random_agent=RandomAgent(
        valuation=my_valuation,
        budget=B,
        T=n_users, 
        eta=eta
    )
    
    
    auction = SecondPriceAuction(ctrs)
    
    # pricing parameters
    
    T=2 # number of days
 
    for t in range(T):
        print(f'Day {t+1}')
        utilities, my_bids, my_payments, total_wins = loop_auction_day(
            auction=auction, 
            agent=random_agent, 
            other_bids=other_bids, 
            n_users=n_users
        )
        print(f'Total Utility: {utilities.sum()}')
        print(f'Mean Utility: {utilities.mean()}')
        print(f'Mean Bid: {my_bids.mean()}')
        print(f'Mean Payment: {my_payments.mean()}')
        print(f'Mean # of Wins: {total_wins/n_users}')
        print('---'*10)
        auction_agent.update_valuation()
        B= 150


    
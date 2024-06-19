#auction
from Auction.SecondPriceAuction import SecondPriceAuction
from Auction.MultiplicativePacingAgent import MultiplicativePacingAgent
import numpy as np
import argparse
from Auction.UCBAgent import UCBAgent

#pricing
from Pricing.GPThompson import GPThompson
from Pricing.GPUCBAgent import GPUCBAgent
from Pricing.GPThompsonContinuous import GPThompsonContinuous
from Pricing.GPUCBAgentContinuous import GPUCBAgentContinuous
from Pricing.StochasticPricingEnvironment import StochasticPricingEnvironment

import numpy as np 

def discretize(T):
    epsilon = T**(-0.33)
    K = int(1/epsilon)
    return K

def rescale(x, min_x, max_x):
    return min_x + (max_x-min_x)*x

def initialize_stoch_auctions(args):
    # noise in the environment
    eta = 1/np.sqrt(args.n_users)
    if args.bidding_agent == 'ucb':
        K_disc = discretize(args.T)
        print('INITIALIZING UCB BIDDING AGENT...')
        adv_agent = UCBAgent(
            valuation=args.valuation,
            budget=args.B,
            n_users=args.n_users,
            T=args.T,
            K=K_disc
            
        )
    else:
        print('INITIALIZING MULTIPLICATIVE BIDDING AGENT...')
        adv_agent = MultiplicativePacingAgent(
            valuation=args.valuation,
            budget=args.B,
            T=args.T, 
            n_users= args.n_users,
            eta=eta
        )
    auction = SecondPriceAuction(args.ctrs)
    return auction, adv_agent
def initialize_stoch_pricing(args):
    K = discretize(args.T)
    prices = np.linspace(args.min_price, args.max_price, K)
    reward_function = lambda price, n_sales: (price-args.cost)*n_sales
    maximum_profit = reward_function(max(prices), args.n_customers)
    # the maximum possible reward is selling at the maximum price to every possible customer

    # conversion prob
    conversion_probability = lambda p: 1-p/20 #TODO: try to change it
            
    # clairvoyant
    profit_curve = reward_function(prices, args.n_customers*conversion_probability(prices))
    best_price_index = np.argmax(profit_curve)
    best_price = prices[best_price_index]
           
           
    expected_clairvoyant_rewards = np.repeat(profit_curve[best_price_index], args.T)
    if args.discretize_price==1:
        
        if args.pricing_agent == 'ucb':
            print('INITIALIZING DISCRETIZED UCB PRICING AGENT...')
            gp_agent = GPUCBAgent(args.T, discretization=K)
        else:
            print('INITIALIZING DISCRETIZED THOMPSON PRICING AGENT...')
            gp_agent = GPThompson(args.T, discretization=K)    
    else:
        if args.pricing_agent == 'ucb':
            print('INITIALIZING CONTINUOUS UCB PRICING AGENT...')
            gp_agent = GPUCBAgentContinuous(args.T)
        else:
            print('INITIALIZING CONTINUOUS THOMPSON PRICING AGENT...')
            gp_agent = GPThompsonContinuous(args.T)
    env = StochasticPricingEnvironment(
        conversion_probability=lambda p: 1-p/20,
        cost=args.cost
    )
    return gp_agent, env, expected_clairvoyant_rewards
    
    
    
    
def loop_auction_day(auction, agent, other_bids,seed, n_users=1000):
    utilities = np.array([])
    my_bids = np.array([])
    my_payments = np.array([])
    total_wins = 0
    m_t = other_bids.max(axis=0)
    
    np.random.seed(seed)
    for u in range(n_users):
        # interaction
        my_bid = agent.bid()
        bids = np.append(my_bid, other_bids[:, u].ravel())
        winners, payments_per_click = auction.round(bids=bids)
        my_win = int(winners==0)
        f_t, c_t = (args.valuation-m_t[u])*my_win, m_t[u]*my_win
        agent.update(f_t, c_t)
        # logging
        utilities = np.append(utilities, f_t)
        my_bids = np.append(my_bids, my_bid)
        my_payments = np.append(my_payments, c_t)
        total_wins+=my_win
    #print(f'Total # of Wins: {total_wins}')
    return utilities, my_bids, my_payments, total_wins

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, help='type of environment: adversarial or stochastic', default='stochastic')
    parser.add_argument('--problem', type=str, help='Play only pricing, only auction or both', default='both')
    parser.add_argument('--advertisers', type=int, help='number of advertisers for auction problem', default=4)
    parser.add_argument('--T', type=int, help='number of days', default=100)
    parser.add_argument('--n_users', type=int, help='number of users', default=1000)
    parser.add_argument('--n_customers', type=int, help='number of customers for pricing problem', default=100)
    parser.add_argument('--B', type=int, help='budget for auction problem', default=15000)
    parser.add_argument('--valuation', type=float, help='valuation for auction problem', default=0.8)
    parser.add_argument('--cost', type=int, help='cost for pricing problem', default=10)
    parser.add_argument('--ctrs', nargs='+', type=float, help='conversion rates for advertisers', default=[1, 1, 1, 1])
    parser.add_argument('--eta', type=float, help='noise in the environment', default=1/np.sqrt(1000))
    parser.add_argument('--min_price', type=int, help='minimum price for pricing problem', default=10)
    parser.add_argument('--max_price', type=int, help='maximum price for pricing problem', default=20)
    parser.add_argument('--pricing_agent', type=str, help="type of agent of the stoch pricing problem", default='ucb')
    parser.add_argument('--bidding_agent', type=str, help="type of agent of the stoch pricing problem", default='mult')
    parser.add_argument('--discretize_price', type=int, help="discretize the price space", default=1)
    #TODO: ADD DISTRIBUTIONS ARGUMENT
    
    # if ctrs is not given or is less than the number of advertisers, fill with ones
    args = parser.parse_args()
    if len(args.ctrs) < args.advertisers:
        args.ctrs += [1]*(args.advertisers-len(args.ctrs))
    return args
if __name__ == '__main__':
    args = parse()
    
    np.random.seed(42)
    #TODO: INITIALIZE DIFFERENT COMBIANATIONS OF ENVIRONMENTS
    
    if args.env == 'stochastic':
        print('INITIALIZING STOCHASTIC ENVIRONMEN...')
        if args.problem == 'auction':
            
            auction, adv_agent = initialize_stoch_auctions(args)
            total_wins_period=0
            for t in range(args.T):
                print(f'Day {t+1}')
                day_seed= np.random.randint(0, 1000)
                print(f'Seed: {day_seed}')
                np.random.seed(day_seed)
                other_bids = np.random.uniform(0, 1, size = (args.advertisers-1, args.n_users))
                utilities, my_bids, my_payments, total_wins = loop_auction_day(
                    auction=auction, 
                    agent=adv_agent, 
                    other_bids=other_bids, 
                    seed=day_seed,
                    n_users=args.n_users
                )
                total_wins_period+=total_wins
                print(f'Total Utility: {utilities.sum()}')
                print(f'Mean Utility: {utilities.mean()}')
                print(f'Mean Bid: {my_bids.mean()}')
                print(f'Mean Payment: {my_payments.mean()}')
                print(f'Mean # of Wins: {total_wins/args.n_users}')
                print(f'Total # of Wins: {total_wins}')
                print('---'*10)
                adv_agent.update_per_round()
            print(f'Total # of Wins: {total_wins_period}')
            print(f'Total Bids: {args.T*args.n_users}')   
            
        elif args.problem == 'pricing':
            gp_agent, env, expected_clairvoyant_rewards = initialize_stoch_pricing(args)
            gp_agent_rewards = np.array([])
            total_revenue = 0
            for t in range(args.T):
                p_t = gp_agent.pull_arm()
                p_t = rescale(p_t, args.min_price, args.max_price)
                d_t, r_t = env.round(p_t, n_t=args.n_customers)
                gp_agent.update(r_t/args.n_customers)
                gp_agent_rewards = np.append(gp_agent_rewards, r_t)
                total_revenue+=r_t
                print(f"day: {t}")
                print(f"price: {p_t}")
                print(f"revenue: {r_t}")
                print("----------------------------")
            print(f'Total Revenue: {total_revenue}')
        elif args.problem == 'both':
            # pricing
            gp_agent, env, expected_clairvoyant_rewards = initialize_stoch_pricing(args)
            gp_agent_rewards = np.array([])
            total_revenue = 0
            
            # auction
            auction, adv_agent = initialize_stoch_auctions(args)
            total_wins_period=0
            for t in range(args.T):
                # run auctions for the day
                print(f'Day {t+1}')
                day_seed= np.random.randint(0, 1000)
                print(f'Seed: {day_seed}')
                np.random.seed(day_seed)
                other_bids = np.random.uniform(0, 1, size = (args.advertisers-1, args.n_users))
                utilities, my_bids, my_payments, total_wins = loop_auction_day(
                    auction=auction, 
                    agent=adv_agent, 
                    other_bids=other_bids, 
                    seed=day_seed,
                    n_users=args.n_users
                )
                total_wins_period+=total_wins
                print(f'AUCTION AT DAY {t+1}')
                print(f'Total Utility: {utilities.sum()}')
                print(f'Mean Utility: {utilities.mean()}')
                print(f'Mean Bid: {my_bids.mean()}')
                print(f'Mean Payment: {my_payments.mean()}')
                print(f'Mean # of Wins: {total_wins/args.n_users}')
                print(f'Total # of Wins: {total_wins}')
                adv_agent.update_per_round()  
                
                
                # run pricing for the day
                p_t = gp_agent.pull_arm()
                p_t = rescale(p_t, args.min_price, args.max_price)
                d_t, r_t = env.round(p_t, n_t=total_wins)
                gp_agent.update(r_t/total_wins)
                gp_agent_rewards = np.append(gp_agent_rewards, r_t)
                total_revenue+=r_t
                print(f"PRICING AT DAY {t+1}")
                print(f"customers: {total_wins}")
                print(f"price: {p_t}")
                print(f"revenue: {r_t}")
                print("----------------------------")  
            print(f'Total # of Wins: {total_wins_period}')
            print(f'Total Revenue: {total_revenue}')             
                          
            
    elif args.env == 'adversarial':
        if args.problem == 'auction':
            pass
        elif args.problem == 'pricing':
            pass
        elif args.problem == 'both':
            pass

 
    



    
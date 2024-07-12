import numpy as np


class PricingSlightlyNonstationaryEnvironment:
    def __init__(self, conversion_probability, cost, variance):
        self.conversion_probability = conversion_probability
        self.cost = cost
        self.variance = variance
        self.t = 0
        self.original_params = (conversion_probability, cost, variance)  # Store the original parameters

    def round(self, p_t, n_t):
        # Calculate probability with slight variation
        probability = self.conversion_probability(p_t, self.t) + np.random.normal(0, self.variance)
        probability = np.clip(probability, 0, 1)

        # Calculate demand and profit
        d_t = np.random.binomial(n_t, probability)
        r_t = (p_t - self.cost) * d_t

        # Increment time step
        self.t += 1

        # Return demand and profit
        return d_t, r_t

    def reset(self):
        # Reinitialize with the original parameters
        self.__init__(*self.original_params)
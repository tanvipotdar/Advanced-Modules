
import numpy as np

# define all parameters
T = 1                           # expiry (hours)
N = 1e4                         # number of shares
mc_paths = 1e4                  # number of monte carlo simulations
s_sigma = 0.001                 # vol of the stock
s0 = 100                        # initial stock price
phi = 1e-3                      # urgency parameter
alpha = 1e-3                    # impatience parameter
k_eta = 0.1                     # mean reversion rate of k
k_mu = 1e-4                     # long running mean of k
k_sigma = 1.5e-3                # vol of k

b_eta = 0.01                    # mean reversion rate of b
b_mu = 1e-4                     # long running mean of b
b_sigma = 3e-3                  # vol of b


def run_almgren_chriss_strategy():
    # time in minutes
    t = T*60
    time = np.zeros(t+1)
    # stochastic process
    stock_price_process = np.zeros(t+1)
    mid_price_process = np.zeros(t+1)
    execution_price_process = np.zeros(t+1)



    gamma = np.sqrt(phi/k_mu)
    zeta = (alpha - 0.5*b_mu + np.sqrt(phi*k_mu))/(alpha - 0.5*b_mu - np.sqrt(phi*k_mu))




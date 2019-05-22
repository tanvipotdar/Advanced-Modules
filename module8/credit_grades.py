import numpy as np
import pandas as pd
from scipy.stats import norm


# constants
# T = 10 # expiry
# N = 10000 # number of monte carlo simulations
#
# s0 = 100 # initial stock price value
# s_ref = 100 # reference stock price value
# s_ref_sigma = 0.35 # reference stock vol
# D = 50 # debt per share
# l = 0.5 # mean of global debt recovery rate/ default barrier
# l_sigma = 0.3 # vol of global debt recovery rate/ default barrier
# r = 0.05 # risk free rate of interest
# R = 0.5 # recovery rate on underlying credit


def g(x, d, r, sigma):
    z = np.sqrt(0.25 + 2 * r/sigma**2)
    a = -np.log(d) / (sigma * np.sqrt(x))
    b = z * sigma * np.sqrt(x)
    param1 = d**(z+1/2) * norm.cdf(a-b)
    param2 = d**(-z+1/2) * norm.cdf(a+b)
    return param1 + param2


def creditgrades(r, t, S0=30, S_ref=30, sigma=0.35, D=10, L=0.5, lmbda=0.3, R=0.5):
    d = (S0 + L*D)/(L*D) * np.exp(lmbda**2)
    eta = lmbda**2 / sigma**2

    a_squared = (sigma*(S_ref/(S_ref+L*D)))**2*t + lmbda**2
    a = np.sqrt(a_squared)
    prob = np.zeros(len(t))
    cds_spread = np.zeros(len(t))

    for i in range(len(t)):
        prob[i] = norm.cdf(-a[i]/2 + np.log(d)/a[i]) - d*norm.cdf(-a[i]/2 - np.log(d)/a[i])

    x1 = np.exp(r*eta)*(g(t+eta,d, r, sigma)-g(t,d, r, sigma))
    x2 = 1 - prob*np.exp(-r*t) - np.exp(r*eta)*(g(t+eta, d, r, sigma)-g(t, d, r, sigma))
    cds_spread = r*(1-R)* (x1/x2)
    cds_spread /= 0.0001
    df = pd.DataFrame(data=[t, prob, cds_spread, sigma]).T
    df.columns = ['maturity', 'survival_prob', 'cds_spread', 'equity_vol']
    return df


# def calculate_survival_probability_and_spread():
#     t = T
#     d = (s0 + l*D)/(l*D) * np.exp(l_sigma**2)
#     eta = l_sigma**2/s_ref_sigma**2

#     asset_price = np.zeros(t+1)
#     prob = np.zeros(t+1)

#     for i in range(t+1):
#         param1 = s_ref_sigma * s_ref/(s_ref + l*D)
#         asset_price[i] = np.sqrt(param1**2*(i/60) + l_sigma**2)
#         prob[i] = norm.cdf(-asset_price[i]/2 + np.log(d)/asset_price[i]) - d*norm.cdf(-asset_price[i]/2 - np.log(d)/asset_price[i])

#     numerator = 1 - prob[0] + np.exp(r*eta)*(g(t+eta, d) - g(eta, d))
#     denominator = prob[0] - prob[t]*np.exp(-r*t) - np.exp(r*eta)*(g(t+eta, d) - g(eta, d))
#     cds_spread = r*(1-R)*(numerator/denominator)
#     return prob[t], cds_spread/0.0001

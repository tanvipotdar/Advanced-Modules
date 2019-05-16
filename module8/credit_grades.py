import numpy as np
import pandas as pd
from scipy.stats import norm


# constants
T = 100 # expiry
N = 10000 # number of monte carlo simulations

s0 = 100 # initial stock price value
s_ref = 100 # reference stock price value
s_ref_sigma = 0.35 # reference stock vol
D = 200 # debt per share
l = 0.5 # mean of global debt recovery rate/ default barrier
l_sigma = 0.3 # vol of global debt recovery rate/ default barrier
r = 0.05 # risk free rate of interest
R = 0.5 # recovery rate on underlying credit


def g(x, d):
    z = np.sqrt(0.25 + 2 * r/s_ref_sigma**2)
    a = -np.log(d)/(s_ref_sigma*np.sqrt(x))
    b = z*s_ref_sigma*np.sqrt(x)
    param1 = d**(z+1/2)*norm.cdf(a-b)
    param2 = d**(-z+1/2)*norm.cdf(a+b)
    return param1 + param2


def calculate_survival_probability_and_spread():
    t = T*60
    d = (s0 + l*D)/(l*D) * np.exp(l_sigma**2)
    eta = l_sigma**2/s_ref_sigma**2

    asset_price = np.zeros(t+1)
    prob = np.zeros(t+1)

    for i in range(t+1):
        param1 = s_ref_sigma * s_ref/(s_ref + l*D)
        asset_price[i] = np.sqrt(param1**2*(i/60) + l_sigma**2)
        prob[i] = norm.cdf(-asset_price[i]/2 + np.log(d)/asset_price[i]) - d*norm.cdf(-asset_price[i]/2 - np.log(d)/asset_price[i])

    import ipdb; ipdb.set_trace()
    numerator = 1 - prob[0] + np.exp(r*eta)*(g(t+eta, d) - g(eta, d))
    denominator = prob[0] - prob[t]*np.exp(-r*t) - np.exp(r*eta)*(g(t+eta, d) - g(eta, d))
    cds_spread = r*(1-R)*(numerator/denominator)
    return prob[t], cds_spread/0.0001
 

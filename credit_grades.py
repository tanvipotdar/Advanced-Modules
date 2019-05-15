import numpy as np
import pandas as pd
from scipy.stats import norm


# constants
T = 5               # expiry
N = 10000           # number of monte carlo simulations

s0 = 100            # initial stock price value
s_ref = 100         # reference stock price value
s_ref_sigma = 0.35  # reference stock vol
D = 10              # debt per share
l = 0.5             # mean of global debt recovery rate/ default barrier
l_sigma = 0.3       # vol of global debt recovery rate/ default barrier
r = 0.03            # risk free rate of interest
R = 0.5             # recovery rate on underlying credit


def g(x, d):
    z = np.sqrt(0.25+(2*r)/s_ref_sigma**2)
    a = -np.log(d)/(s_ref_sigma*np.sqrt(x))
    b = z*s_ref_sigma*np.sqrt(x)
    param1 = d**(z+1/2)*norm.cdf(a-b)
    param2 = d**(-z+1/2)*norm.cdf(a+b)
    return param1 + param2


def calculate_survival_probability():
    t = T*60
    d = (s0 + l*D)/l*D) * np.exp(l_sigma**2)
    eta = l_sigma**2/s_ref_sigma**2

    a = np.zeroes(t+1)
    p = np.zeroes(t+1)

    for i in range(t+1):
        param1 = s_ref_sigma * s_ref/(s_ref + l_sigma*D)
        a[i] = np.sqrt(param1**2*(i/60) + l_sigma**2)
        p[i] = norm.cdf(-a[i]/2 + np.log(d)/a[i]) - d*norm.cdf(-a[i]/2 - np.log(d)/a[i])

    numerator = 1 - p[0] + np.exp(r*eta)*(g(t+eta, d) - g(eta, d))
    denominator = p[0] - p[T]*np.exp(-r*t) - np.exp(r*eta)*(g(t+eta, d) - g(eta, d))
    cds_spread = r*(1-R)*(numerator/denominator)
    return p[T], cds_spread

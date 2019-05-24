import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_survival_probability(L_mean, S0, S_ref, sigma_ref, D, lmbda, t):
    d = (S0 + L_mean*D)/(L_mean*D) * np.exp(lmbda**2)
    a_squared = (sigma_ref*S_ref/(S_ref+L_mean*D))**2*t + lmbda**2
    a_t = np.sqrt(a_squared)
    prob = norm.cdf(-a_t/2 + np.log(d)/a_t) - d*norm.cdf(-a_t/2 - np.log(d)/a_t)
    return prob


def G(u, d, sigma_at, r):
    z = np.sqrt(0.25 + 2 * r/sigma_at**2)

    a = -np.log(d) / (sigma_at * np.sqrt(u))
    b = z * sigma_at * np.sqrt(u)

    param1 = d**(z+0.5) * norm.cdf(a-b)
    param2 = d**(-z+0.5) * norm.cdf(a+b)
    return param1 + param2


def creditgrades(L_mean, S0, S_ref, sigma_ref, D, lmbda, r, R, t):
    prob_0 = calculate_survival_probability(L_mean, S0, S_ref, sigma_ref, D, lmbda, 0)
    prob_t = calculate_survival_probability(L_mean, S0, S_ref, sigma_ref, D, lmbda, t)

    d = (S0 + L_mean*D)/(L_mean*D) * np.exp(lmbda**2)
    sigma_at = sigma_ref*S_ref/(S_ref+L_mean*D) 
    eta = lmbda**2 / sigma_at**2

    x1 = np.exp(r*eta) * (G(t+eta, d, sigma_at, r) - G(eta, d, sigma_at, r))
    x2 = 1 - prob_0 + x1
    x3 = prob_0 - prob_t*np.exp(-r*t) - x1


    cds_spread = r*(1-R)*(x2/x3)*10000
    df = pd.DataFrame(data=[t, sigma_ref, S0, D, cds_spread, prob_t]).T
    df.columns = ['maturity', 'equity_vol', 'S0', 'debt', 'cds_spread', 'survival_probability']
    # return cds_spread, prob_t
    return df

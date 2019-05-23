import numpy as np
import pandas as pd


def calculate_survival_probability(L_mean, S0, S_ref, sigma_ref, D, lmbda, t):
    d = (S0 + L_mean*D)/(L_mean*D) * np.exp(lmbda**2)
    a_squared = (sigma*S_ref/(S_ref+L_mean*D))**2*t + lmbda**2
    a_t = np.sqrt(a_squared)
    prob = norm.cdf(-a[i]/2 + np.log(d)/a[i]) - d*norm.cdf(-a[i]/2 - np.log(d)/a[i])
    return prob


def G(u, d, sigma_at, r):
    z = np.sqrt(0.25 + 2 * r/sigma_at**2)

    a = -np.log(d) / (sigma_at * np.sqrt(u))
    b = z * sigma_at * np.sqrt(u)

    param1 = d**(z+0.5) * norm.cdf(a-b)
    param2 = d**(-z+0.5) * norm.cdf(a+b)
    return param1 + param2


def calculate_cds_spread(L_mean, S0, S_ref, sigma_ref, D, lmbda, r, R, t):
    prob_0 = calculate_survival_probability(L_mean, S0, S-ref, sigma_ref, D, lmbda, 0)
    prob_t = calculate_survival_probability(L_mean, S0, S-ref, sigma_ref, D, lmbda, t)

    sigma_at = sigma*S_ref/(S_ref+L_mean*D)
    eta = lmbda**2 / sigma**2

    x1 = np.exp(r*eta) * (G(t+eta, d, sigma_at, r) - G(eta, d. sigma_at, r))
    x2 = 1 - prob_0 + x1
    x3 = prob_0 - prob_t*np.exp(-r*t) - x1

    cds_spread = r*(1-R)*(x2/x3)*10000
    return cds_spread

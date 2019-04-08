import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# define all parameters
T = 1  # expiry (hours)
N = 1e4  # number of shares
s_sigma = 0.015  # vol of the stock
s0 = 195.35  # initial stock price
phi = 1e-3  # urgency parameter
alpha = 1e-3  # terminal liquidation penalty

k_eta = 1  # mean reversion rate of k
k_mu = 1e-3  # long running mean of k
k_sigma = 0.01  # vol of k

b_eta = 1  # mean reversion rate of b
b_mu = 2e-3  # long running mean of b
b_sigma = 0.01  # vol of b


def run_almgren_chriss_with_constant_pi(alpha=alpha, phi=phi, T=T, k_mu=k_mu, b_mu=b_mu, s_sigma=s_sigma):
    # time in minutes
    t = T * 60
    dt = 1 / 60.

    # stochastic processes
    w = np.random.randn(t + 1)  # brownian motion for stock price, normal random variables
    s = np.zeros(t + 1)  # mid-price process
    e = np.zeros(t + 1)  # execution price process
    v = np.zeros(t + 1)  # trading speed process
    x = np.zeros(t + 1)  # income process
    q = np.zeros(t + 1)  # inventory process

    gamma = np.sqrt(phi / k_mu)
    zeta = (alpha - 0.5 * b_mu + np.sqrt(phi * k_mu)) / (alpha - 0.5 * b_mu - np.sqrt(phi * k_mu))

    for i in range(t + 1):
        q[i] = N if i == 0 else q[i - 1] - min(v[i - 1], q[i - 1])
        s[i] = s0 if i == 0 else max(s[i - 1] + (- b_mu * v[i - 1] * dt + s_sigma * w[i] * np.sqrt(i * dt)), 0)

        tT = (t - i) / float(t)
        v_num = zeta * np.exp(gamma * tT) + np.exp(-gamma * tT)
        v_denom = zeta * np.exp(gamma * tT) - np.exp(-gamma * tT)
        v[i] = gamma * (v_num / v_denom) * q[i]
        v[i] = max(v[i], 0) * dt
        e[i] = max(s[i] - k_mu * v[i], 0)
        if i == t:
            x[i] = 0
            q[i] = q[i - 1]
        else:
            x[i] = e[i] * min(v[i], q[i])

    x[t] = q[t] * (s[t] - alpha * q[t])
    return x.cumsum(), q, v


def calculate_stochastic_path(mu, eta, sigma):
    t = T * 60
    dt = 1/60.
    w = np.random.randn(t + 1)  # brownian motion for the price impact param
    pi_path = np.zeros(t + 1)  # path for the price impact param
    pi_path[0] = mu

    for i in range(1, t + 1):
        pi_path[i] = pi_path[i - 1] + (eta * mu - eta * pi_path[i - 1]) * dt + sigma * w[i] * np.sqrt(pi_path[i-1] * dt)
    return pi_path


def run_almgren_chriss_with_stochastic_pi(alpha=alpha, phi=phi, T=T, k_mu=k_mu, k_eta=k_eta, k_sigma=k_sigma, b_mu=b_mu, b_eta=b_eta, b_sigma=b_sigma,
                                          s_sigma=s_sigma):
    # time in minutes
    t = T * 60
    dt = 1 / 60.

    # stochastic processes
    w = np.random.randn(t + 1)  # brownian motion for stock price, normal random variables
    s = np.zeros(t + 1)  # mid-price process
    e = np.zeros(t + 1)  # execution price process
    v = np.zeros(t + 1)  # trading speed process
    x = np.zeros(t + 1)  # income process
    q = np.zeros(t + 1)  # inventory process

    # coefficients for optimal speed
    gamma = np.zeros(t + 1)
    zeta = np.zeros(t + 1)

    k = calculate_stochastic_path(k_mu, k_eta, k_sigma)
    b = calculate_stochastic_path(b_mu, b_eta, b_sigma)

    for i in range(t + 1):
        gamma[i] = np.sqrt(phi / k[i])
        zeta[i] = (alpha - 0.5 * b[i] + np.sqrt(phi * k[i])) / (alpha - 0.5 * b[i] - np.sqrt(phi * k[i]))

        q[i] = N if i == 0 else q[i - 1] - min(v[i - 1], q[i - 1])
        s[i] = s0 if i == 0 else max(s[i - 1] - b[i] * v[i - 1] * dt + s_sigma * w[i] * np.sqrt(i * dt), 0)

        tT = (t - i) / float(t)
        v_num = zeta[i] * np.exp(gamma[i] * tT) + np.exp(-gamma[i] * tT)
        v_denom = zeta[i] * np.exp(gamma[i] * tT) - np.exp(-gamma[i] * tT)
        v[i] = gamma[i] * (v_num / v_denom) * q[i]
        v[i] = max(v[i], 0) * dt
        e[i] = max(s[i] - k[i] * v[i], 0)
        if i == t:
            x[i] = 0
            q[i] = q[i - 1]
        else:
            x[i] = e[i] * min(v[i], q[i])

    x[t] = q[t] * (s[t] - alpha * q[t])
    return x.cumsum(), q, v

def calculate_performance(mc_paths, alpha=alpha, phi=phi, T=T, k_mu=k_mu, k_eta=k_eta, k_sigma=k_sigma, b_mu=b_mu, b_eta=b_eta, b_sigma=b_sigma,
                          s_sigma=s_sigma):
    cash_from_constant_strategy = np.zeros(mc_paths)
    cash_from_stochastic_strategy = np.zeros(mc_paths)
    performance = np.zeros(mc_paths)

    for x in range(mc_paths):
        cash_from_constant_strategy[x] = run_almgren_chriss_with_constant_pi(alpha=alpha, phi=phi, T=T, k_mu=k_mu, b_mu=b_mu, s_sigma=s_sigma)[0][-1]
        cash_from_stochastic_strategy[x] = run_almgren_chriss_with_stochastic_pi(alpha=alpha, phi=phi, T=T, k_mu=k_mu, k_eta=k_eta, k_sigma=k_sigma,
        b_mu=b_mu, b_eta=b_eta, b_sigma=b_sigma, s_sigma=s_sigma)[0][-1]
        performance[x] = (cash_from_stochastic_strategy[x] - cash_from_constant_strategy[x]) / cash_from_constant_strategy[x]
        performance[x] *= 10000

    return performance

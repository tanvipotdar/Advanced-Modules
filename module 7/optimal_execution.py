
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


# define all parameters
T = 1  # expiry (hours)
N = 1e4  # number of shares
s_sigma = 0.027  # vol of the stock
s0 = 100  # initial stock price
phi = 1e-3  # running inventory penalty
alpha = 1e-3  # terminal liquidation penalty

k_eta = 10  # mean reversion rate of k
k_mu = 1e-3  # long running mean of k
k_sigma = 1.5e-3  # vol of k

b_eta = 10  # mean reversion rate of b
b_mu = 1e-3  # long running mean of b
b_sigma = 3e-3  # vol of b


def run_almgren_chriss_with_constant_pi(alpha=alpha, phi=phi, T=T, k_mu=k_mu, b_mu=b_mu, s_sigma=s_sigma):
    # time in minutes
    t = T * 60

    # stochastic processes
    w = np.random.randn(t + 1)  # brownian motion for stock price, normal random variables
    s = np.zeros(t + 1)  # mid-price process
    e = np.zeros(t + 1)  # execution price process
    v = np.zeros(t + 1)  # trading speed process
    x = np.zeros(t + 1)  # income process
    q = np.zeros(t + 1)  # inventory process

    q[0] = N
    gamma = np.sqrt(phi / k_mu)
    zeta = (alpha - 0.5 * b_mu + np.sqrt(phi * k_mu)) / (alpha - 0.5 * b_mu - np.sqrt(phi * k_mu))

    for i in range(t + 1):
        q[i] = N if i == 0 else q[i - 1] - min(v[i - 1], q[i-1])
        s[i] = s0 if i == 0 else s[i - 1] - (b_mu * v[i - 1] / 60. + s_sigma * w[i])

        tT = (t - i) / float(t)
        v_num = zeta * np.exp(gamma * tT) + np.exp(-gamma * tT)
        v_denom = zeta * np.exp(gamma * tT) - np.exp(-gamma * tT)
        v[i] = gamma * (v_num / v_denom) * q[i]
        # divide by 60 as dt=1/60
        v[i] = max(v[i], 0) / 60.
        e[i] = s[i] - k_mu * v[i]
        if i == t:
            x[i] = 0
            q[i] = q[i - 1]
        else:
            x[i] = e[i] * min(v[i], q[i])

    x[t] = q[t] * (s[t] - alpha * q[t])
    return x.cumsum(), q


def calculate_stochastic_path(mu, eta, sigma):
    t = T * 60
    w = np.random.randn(t + 1)  # brownian motion for the price impact param
    pi_path = np.zeros(t + 1)  # path for the price impact param
    pi_path[0] = mu

    for i in range(1, t + 1):
        # tT = (t - i) / t
        # pi_path[i] = mu if i == 0 else mu + sigma * np.sqrt((np.exp(-2 * eta * tT)) / (2 * eta)) * w[i] * np.sqrt(i/60)
        pi_path[i] = pi_path[i - 1] + (eta * mu - eta * pi_path[i - 1]) / 60. + sigma * w[i] * np.sqrt((i * pi_path[i - 1]) / 60.)
    return pi_path


def run_almgren_chriss_with_stochastic_pi(alpha=alpha, phi=phi, T=T, k_mu=k_mu, k_eta=k_eta, k_sigma=k_sigma, 
	b_mu=b_mu, b_eta=b_eta, b_sigma=b_sigma, s_sigma=s_sigma):
    # time in minutes
    t = T * 60

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

    q[0] = N
    k = calculate_stochastic_path(k_mu, k_eta, k_sigma)
    b = calculate_stochastic_path(b_mu, b_eta, b_sigma)

    for i in range(t + 1):
        gamma[i] = np.sqrt(phi / k[i])
        zeta[i] = (alpha - 0.5 * b[i] + np.sqrt(phi * k[i])) / (alpha - 0.5 * b[i] - np.sqrt(phi * k[i]))

        q[i] = N if i == 0 else q[i - 1] - v[i - 1]
        s[i] = s0 if i == 0 else s[i - 1] - (b[i] * v[i - 1] / 60 + s_sigma * w[i] * np.sqrt(i / 60.))

        tT = (t - i) / float(t)
        v_num = zeta[i] * np.exp(gamma[i] * tT) + np.exp(-gamma[i] * tT)
        v_denom = zeta[i] * np.exp(gamma[i] * tT) - np.exp(-gamma[i] * tT)
        v[i] = gamma[i] * (v_num / v_denom) * q[i]
        v[i] = max(v[i], 0) / 60.
        e[i] = s[i] - k[i] * v[i]
        if i == t:
            x[i] = 0
            q[i] = q[i - 1]
        else:
            x[i] = e[i] * min(v[i], q[i])

    x[t] = q[t] * (s[t] - alpha * q[t])
    return x.cumsum(), q


def calculate_performance(mc_paths):
    cash_from_constant_strategy = np.zeros(mc_paths)
    cash_from_stochastic_strategy = np.zeros(mc_paths)
    performance = np.zeros(mc_paths)

    for x in range(mc_paths):
        cash_from_constant_strategy[x] = run_almgren_chriss_with_constant_pi()[0][-1]
        cash_from_stochastic_strategy[x] = run_almgren_chriss_with_stochastic_pi()[0][-1]
        performance[x] = (cash_from_stochastic_strategy[x] - cash_from_constant_strategy[x]) / cash_from_constant_strategy[x]
        performance[x] *= 10000


    return performance


def plot_all():
	# histogram of performance
	histogram, ax = plt.subplots(figsize=(10,5))
	ax.set(ylabel='Frequency', xlabel='Performance (bps)', title='Performance of stochastic vs constant AC')
	histogram = plt.hist(a, density=False)
	plt.show()

	# plot inventory process with different alphas
	a0001,b0001 = run_almgren_chriss_with_constant_pi(alpha=0.0001)
	a001,b001 = run_almgren_chriss_with_constant_pi(alpha=0.001)
	a01,b01=run_almgren_chriss_with_constant_pi(alpha=0.01)
	a1,b1=run_almgren_chriss_with_constant_pi(alpha=0.1)

	f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col')
	ax1.set(ylabel='Inventory', xlabel='time (minutes)')
	ax1.plot(b0001, label='alpha=0.0001')
	ax1.plot(b001, label='alpha=0.001')
	ax1.plot(b01, label='alpha=0.01')
	ax1.plot(b1, label='alpha=0.1')
	ax1.legend(loc="upper right")

	ax2.set(ylabel='Income earned', xlabel='time (minutes)')
	ax2.plot(a0001[:-1], label='alpha=0.0001')
	ax2.plot(a001[:-1], label='alpha=0.001')
	ax2.plot(a01[:-1], label='alpha=0.01')
	ax2.plot(a1[:-1], label='alpha=0.1')
	ax2.legend(loc="upper left")

	# plot inventory process with different phis
	a0001,b0001 = run_almgren_chriss_with_constant_pi(phi=0.0001)
	a001,b001 = run_almgren_chriss_with_constant_pi(phi=0.001)
	a01,b01=run_almgren_chriss_with_constant_pi(phi=0.01)
	a1,b1=run_almgren_chriss_with_constant_pi(phi=0.1)

	f, ((ax1, ax2)) = plt.subplots(1, 2, sharex='col')
	ax1.set(ylabel='Inventory', xlabel='time (minutes)')
	ax1.plot(b0001, label='phi=0.0001')
	ax1.plot(b001, label='phi=0.001')
	ax1.plot(b01, label='phi=0.01')
	ax1.plot(b1, label='phi=0.1')
	ax1.legend(loc="upper right")

	ax2.set(ylabel='Income earned', xlabel='time (minutes)')
	ax2.plot(a0001[:-1], label='phi=0.0001')
	ax2.plot(a001[:-1], label='phi=0.001')
	ax2.plot(a01[:-1], label='phi=0.01')
	ax2.plot(a1[:-1], label='phi=0.1')
	ax2.legend(loc="upper left")

	# plot 





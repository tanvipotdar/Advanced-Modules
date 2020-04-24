import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# define all parameters
T = 1  # expiry (hours)
N = 1e4  # number of shares
s_sigma = 0.015  # vol of the stock
s0 = 195.35  # initial stock price
phi = 1e-3  # running inventory penalty
alpha = 1e-3  # terminal liquidation penalty

k_eta = .1  # mean reversion rate of k
k_mu = 1e-3  # long running mean of k
k_sigma = 0.01  # vol of k

b_eta = .1  # mean reversion rate of b
b_mu = 1e-3  # long running mean of b
b_sigma = 0.02  # vol of b


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

    gamma = np.sqrt(phi / k_mu)
    zeta = (alpha - 0.5 * b_mu + np.sqrt(phi * k_mu)) / (alpha - 0.5 * b_mu - np.sqrt(phi * k_mu))

    for i in range(t + 1):
        q[i] = N if i == 0 else q[i - 1] - min(v[i - 1], q[i - 1])
        s[i] = s0 if i == 0 else max(s[i - 1] + (- b_mu * v[i - 1] / 60. + s_sigma * w[i] * np.sqrt(i / 60.)), 0)

        tT = (t - i) / float(t)
        v_num = zeta * np.exp(gamma * tT) + np.exp(-gamma * tT)
        v_denom = zeta * np.exp(gamma * tT) - np.exp(-gamma * tT)
        v[i] = gamma * (v_num / v_denom) * q[i]
        # divide by 60 as dt=1/60
        v[i] = max(v[i], 0) / 60.
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
    w = np.random.randn(t + 1)  # brownian motion for the price impact param
    pi_path = np.zeros(t + 1)  # path for the price impact param
    pi_path[0] = mu

    for i in range(1, t + 1):
        pi_path[i] = pi_path[i - 1] + (eta * mu - eta * pi_path[i - 1]) / 60. + sigma * w[i] * np.sqrt(pi_path[i-1]/60)
    return pi_path


def run_almgren_chriss_with_stochastic_pi(alpha=alpha, phi=phi, T=T, k_mu=k_mu, k_eta=k_eta, k_sigma=k_sigma, b_mu=b_mu, b_eta=b_eta, b_sigma=b_sigma,
                                          s_sigma=s_sigma):
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

    k = calculate_stochastic_path(k_mu, k_eta, k_sigma)
    b = calculate_stochastic_path(b_mu, b_eta, b_sigma)

    for i in range(t + 1):
        gamma[i] = np.sqrt(phi / k[i])
        zeta[i] = (alpha - 0.5 * b[i] + np.sqrt(phi * k[i])) / (alpha - 0.5 * b[i] - np.sqrt(phi * k[i]))

        q[i] = N if i == 0 else q[i - 1] - min(v[i - 1], q[i - 1])
        s[i] = s0 if i == 0 else max(s[i - 1] - b[i] * v[i - 1] / 60 + s_sigma * w[i] * np.sqrt(i / 60.), 0)

        tT = (t - i) / float(t)
        v_num = zeta[i] * np.exp(gamma[i] * tT) + np.exp(-gamma[i] * tT)
        v_denom = zeta[i] * np.exp(gamma[i] * tT) - np.exp(-gamma[i] * tT)
        v[i] = gamma[i] * (v_num / v_denom) * q[i]
        v[i] = max(v[i], 0) / 60.
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


def plot_income_and_inventory_against_cir_params(low_b, low_k, high_b, high_k, k_low, k_high, b_low, b_high, param_k, param_b):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col')
    ax1.set(ylabel='Income', xlabel='time (minutes)')
    ax1.plot(k_low[0][:-1], label='{}={}'.format(param_k, low_k))
    ax1.plot(k_high[0][:-1], label='{}={}'.format(param_k, high_k))
    ax1.legend(loc="upper right")

    ax2.set(ylabel='Inventory', xlabel='time (minutes)')
    ax2.plot(k_low[1], label='{}={}'.format(param_k, low_k))
    ax2.plot(k_high[1], label='{}={}'.format(param_k, high_k))
    ax2.legend(loc="upper right")

    ax3.set(ylabel='Income', xlabel='time (minutes)')
    ax3.plot(b_low[0][:-1], label='{}={}'.format(param_b, low_b))
    ax3.plot(b_high[0][:-1], label='{}={}'.format(param_b, high_b))
    ax3.legend(loc="upper right")

    ax4.set(ylabel='Inventory', xlabel='time (minutes)')
    ax4.plot(b_low[1], label='{}={}'.format(param_b, low_b))
    ax4.plot(b_high[1], label='{}={}'.format(param_b, high_b))
    ax4.legend(loc="upper right")
    plt.tight_layout()


def plot_all():
    # histogram of performance
    histogram, ax = plt.subplots(figsize=(10, 5))
    ax.set(ylabel='Frequency', xlabel='Performance (bps)')
    a = calculate_performance(10000)
    histogram = plt.hist(a, density=False)

    # plot income and inventory process with different alphas
    a0001, b0001 = run_almgren_chriss_with_constant_pi(alpha=0.0001)
    a001, b001 = run_almgren_chriss_with_constant_pi(alpha=0.001)
    a01, b01 = run_almgren_chriss_with_constant_pi(alpha=0.01)
    a1, b1 = run_almgren_chriss_with_constant_pi(alpha=0.1)

    f, (ax1, ax2) = plt.subplots(1, 2, sharex='col')
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
    plt.tight_layout()

    # plot income and inventory process with different phis
    a0001, b0001 = run_almgren_chriss_with_constant_pi(phi=0.0001)
    a001, b001 = run_almgren_chriss_with_constant_pi(phi=0.001)
    a01, b01 = run_almgren_chriss_with_constant_pi(phi=0.01)
    a1, b1 = run_almgren_chriss_with_constant_pi(phi=0.1)

    f, (ax1, ax2) = plt.subplots(1, 2, sharex='col')
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
    plt.tight_layout()

    # plot path of CIR process with different eta, sigma values
    low_eta = 0.00001
    high_eta = 100
    etahigh = calculate_stochastic_path(k_mu, high_eta, k_sigma)
    etalow = calculate_stochastic_path(k_mu, low_eta, k_sigma)

    low_vol = 0.00001
    high_vol = 0.1
    volhigh = calculate_stochastic_path(k_mu, k_eta, high_vol)
    vollow = calculate_stochastic_path(k_mu, k_eta, low_vol)

    f, (ax2, ax3) = plt.subplots(1, 2, sharex='col', figsize=(8,3))
    ax2.set(ylabel='Path of CIR process', xlabel='time (minutes)')
    ax2.plot(etalow, label='eta={}'.format(low_eta))
    ax2.plot(etahigh, label='eta={}'.format(high_eta))
    ax2.legend(loc="upper right")

    ax3.set(ylabel='Path of CIR process', xlabel='time (minutes)')
    ax3.plot(vollow, label='sigma={}'.format(low_vol))
    ax3.plot(volhigh, label='sigma={}'.format(high_vol))
    ax3.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    # plot income with different mu values
    low_kmu = 0.00001
    high_kmu = 1000
    kmu_low = run_almgren_chriss_with_constant_pi(k_mu=low_kmu)
    kmu_high = run_almgren_chriss_with_constant_pi(k_mu=high_kmu)

    low_bmu = 0.00001
    high_bmu = 1000
    bmu_low = run_almgren_chriss_with_constant_pi(b_mu=low_bmu)
    bmu_high = run_almgren_chriss_with_constant_pi(b_mu=high_bmu)
    plot_income_and_inventory_against_cir_params(low_bmu, low_kmu, high_bmu, high_kmu, kmu_low, kmu_high, bmu_low, bmu_high, 'k_mu', 'b_mu')

    # plot stochastic ac income with different values of eta and vol
    low_keta = 0.00001
    high_keta = 1000
    k_low = run_almgren_chriss_with_stochastic_pi(k_sigma=low_keta)
    k_high = run_almgren_chriss_with_stochastic_pi(k_sigma=high_keta)

    low_beta = 0.00001
    high_beta = 1000
    b_low = run_almgren_chriss_with_stochastic_pi(b_sigma=low_beta)
    b_high = run_almgren_chriss_with_stochastic_pi(b_sigma=high_beta)
    plot_income_and_inventory_against_cir_params(low_beta, low_keta, high_beta, high_keta, k_low, k_high, b_low, b_high, 'k_eta', 'b_eta')

    # plot outperformance against alpha values
    # alpha_values = range()
    # calculate_performance(10000, alpha=0.0001)
    # calculate_performance(10000, alpha=0.1)

    # plot trading speed and inventory in stochastic vs constant AC
    cx, cq, cv = run_almgren_chriss_with_constant_pi()
    sx, sq, sv = run_almgren_chriss_with_stochastic_pi()
    f, (ax2, ax3, ax4) = plt.subplots(1, 3, sharex='col', figsize=(8, 3))
    ax2.set(ylabel='Trading Speed', xlabel='time (minutes)')
    ax2.plot(cv, label='Constant')
    ax2.plot(sv, label='Stochastic')
    ax2.legend(loc="upper right")

    ax3.set(ylabel='Inventory difference', xlabel='time (minutes)')
    ax3.plot(cq-sq)
    ax4.set(ylabel='Income difference', xlabel='time (minutes)')
    ax4.plot(cx[:-1]-sx[:-1])
    plt.tight_layout()
    plt.show()
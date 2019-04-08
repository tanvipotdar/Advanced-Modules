from optimal_execution import *

def plot_performance_histogram():
    f, ax = plt.subplots(figsize=(8, 5))
    ax.set(ylabel='Frequency', xlabel='Performance (bps)')
    a = calculate_performance(10000)
    plt.hist(a, density=False, bins=20, edgecolor='k')
    plt.axvline(a.mean(), color='k', linestyle='dashed', linewidth=1)
    f.tight_layout()
    plt.show()


def plot_midprice():
    f, ax = plt.subplots(figsize=(5,3))
    ax.set(ylabel='Stock Price', xlabel='time (minutes)')
    plt.plot(ec, label='Constant')
    plt.plot(es, label='Stochastic')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_midprice_vs_execution():
    f, (ax1, ax2) = plt.subplots(1, 2, sharex='col', figsize=(8,3))
    ax1.set(ylabel='Stock Price', xlabel='time (minutes)')
    ax1.plot(ss, label='Mid-Price')
    ax1.plot(es, label='Execution Price')
    ax1.legend()

    ax2.set(ylabel='Stock Price', xlabel='time (minutes)')
    ax2.plot(sc, label='Mid-Price')
    ax2.plot(ec, label='Execution Price')
    ax2.legend()

    plt.tight_layout()
    plt.show()


def plot_x_and_q_vs_alpha():
    a0001, b0001, c0001= run_almgren_chriss_with_constant_pi(alpha=0.0001)
    a001, b001, c001 = run_almgren_chriss_with_constant_pi(alpha=0.001)
    a01, b01, c01 = run_almgren_chriss_with_constant_pi(alpha=0.01)
    a1, b1, c1 = run_almgren_chriss_with_constant_pi(alpha=0.1)

    f, (ax1,ax3) = plt.subplots(1, 2, sharex='col', figsize=(8,3))
    ax1.set(ylabel='Inventory', xlabel='time (minutes)')
    ax1.plot(b0001, label='0.0001')
    ax1.plot(b001, label='0.001')
    ax1.plot(b01, label='0.01')
    ax1.plot(b1, label='0.1')
    ax1.legend(loc="upper right")

    ax3.set(ylabel='Trading Speed', xlabel='time (minutes)')
    ax3.plot(c0001, label='0.0001')
    ax3.plot(c001, label='0.001')
    ax3.plot(c01, label='0.01')
    ax3.plot(c1, label='0.1')
    ax3.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_x_and_q_vs_phi():
    a0001, b0001, c0001 = run_almgren_chriss_with_constant_pi(phi=0.0001)
    a001, b001, c001 = run_almgren_chriss_with_constant_pi(phi=0.001)
    a01, b01, c01 = run_almgren_chriss_with_constant_pi(phi=0.01)
    a1, b1, c1 = run_almgren_chriss_with_constant_pi(phi=0.1)

    f, (ax1, ax3) = plt.subplots(1, 2, sharex='col', figsize=(8,3))
    ax1.set(ylabel='Inventory', xlabel='time (minutes)')
    ax1.plot(b0001, label='0.0001')
    ax1.plot(b001, label='0.001')
    ax1.plot(b01, label='0.01')
    ax1.plot(b1, label='0.1')
    ax1.legend(loc="upper right")

    ax3.set(ylabel='Trading Speed', xlabel='time (minutes)')
    ax3.plot(c0001, label='0.0001')
    ax3.plot(c001, label='0.001')
    ax3.plot(c01, label='0.01')
    ax3.plot(c1, label='0.1')
    ax3.legend(loc="upper right")
    plt.tight_layout()


def plot_CIR_path():
    low_eta = 0.00001
    high_eta = 100
    etahigh = calculate_stochastic_path(k_mu, high_eta, k_sigma)
    etalow = calculate_stochastic_path(k_mu, low_eta, k_sigma)

    low_vol = 0.00001
    high_vol = 0.1
    volhigh = calculate_stochastic_path(k_mu, 1, high_vol)
    vollow = calculate_stochastic_path(k_mu, 1, low_vol)

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


def plot_x_and_q_in_both_models():
    cx, cq, cv = run_almgren_chriss_with_constant_pi()
    sx, sq, sv = run_almgren_chriss_with_stochastic_pi()
    f, (ax2,ax3) = plt.subplots(1,2, sharex='col', figsize=(8, 3))
    ax2.set(ylabel='Trading Speed', xlabel='time (minutes)')
    ax2.plot(cv, label='Constant')
    ax2.plot(sv, label='Stochastic')
    ax2.legend(loc="upper right")

    ax3.set(ylabel='Income', xlabel='time (minutes)')
    ax3.plot(cx[:-1], label='Constant')
    ax3.plot(sx[:-1], label='Stochastic')
    ax3.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def trading_speeds_against_variable():
    cx, cq, cv = run_almgren_chriss_with_constant_pi(phi=1e-1)
    cx1, cq1, cv1 = run_almgren_chriss_with_constant_pi(phi=1e-5)
    sx, sq, sv = run_almgren_chriss_with_stochastic_pi(phi=1e-1)
    sx1, sq1, sv1 = run_almgren_chriss_with_stochastic_pi(phi=1e-5)
    f, (ax2,ax3) = plt.subplots(1,2, sharex='col', figsize=(8, 3))
    ax2.set(ylabel='Trading Speed', xlabel='time (minutes)')
    ax2.plot(cv, label='Constant')
    ax2.plot(sv, label='Stochastic')
    ax2.legend(loc="upper right")

    ax3.set(ylabel='Trading Speed', xlabel='time (minutes)')
    ax3.plot(cv1, label='Constant')
    ax3.plot(sv1, label='Stochastic')
    ax3.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_performance_vs_alpha_phi():
    powers = [10**x for x in range(-6,1)]
    alpha_values = [calculate_performance(100, alpha=x).mean() for x in powers]
    phi_values = [calculate_performance(100, phi=x).mean() for x in powers]

    f, (ax2, ax3) = plt.subplots(1, 2, sharex='col', figsize=(8,3))
    ax2.set(ylabel='Performance (bps)', xlabel='log of alpha')
    ax2.plot(np.log10(powers), np.abs(alpha_values))
    ax3.set(ylabel='Performance (bps)', xlabel='log of phi')
    ax3.plot(np.log10(powers), np.abs(phi_values))
    plt.tight_layout()
    plt.show()


def plot_performance_vs_cir_params():
    # sigma generated with keta as 1
    powers = [10**x for x in range(-6,-1)]
    eta_vals = [calculate_performance(100, b_mu=x, b_sigma=0.0001).mean() for x in powers]
    # bmu_vals = [calculate_performance(1000, b_sigma=x, b_eta=1).mean() for x in powers]
    sigma_vals = [calculate_performance(100, k_mu=x, k_sigma=0.0001).mean() for x in powers]
    f, (ax2, ax3) = plt.subplots(1, 2, sharex='col', figsize=(8,3))
    ax2.set(ylabel='Performance (bps)', xlabel='log of b_mu')
    ax2.plot(np.log10(powers), eta_vals)
    ax3.set(ylabel='Performance (bps)', xlabel='log of k_mu')
    ax3.plot(np.log10(powers), sigma_vals)
    plt.tight_layout()
    plt.show()


def plot_income_against_CIR():
    low_kmu = 0.000001
    kmu_med=0.001
    high_kmu = 100
    kmu_low = run_almgren_chriss_with_constant_pi(k_mu=low_kmu)
    kmu = run_almgren_chriss_with_constant_pi(k_mu=kmu_med)
    kmu_high = run_almgren_chriss_with_constant_pi(k_mu=high_kmu)

    low_bmu = 0.000001
    bmu_med=0.001
    high_bmu = 100
    bmu_low = run_almgren_chriss_with_constant_pi(b_mu=low_bmu)
    bmu = run_almgren_chriss_with_constant_pi(b_mu=bmu_med)
    bmu_high = run_almgren_chriss_with_constant_pi(b_mu=high_bmu)
    plot_income_and_inventory_against_cir_params(low_bmu, low_kmu, kmu_med, high_bmu, high_kmu, bmu_med, kmu_low, kmu_high, kmu, bmu_low, bmu_high, bmu, 'k_mu', 'b_mu')


def plot_income_and_inventory_against_cir_params(low_b, low_k, k_med, high_b, high_k, b_med, k_low, k_high, kmu, b_low, b_high, bmu, param_k, param_b):
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col')
    ax1.set(ylabel='Income', xlabel='time (minutes)')
    ax1.plot(k_low[0][:-1], label='{}={}'.format(param_k, low_k))
    # ax1.plot(kmu[0][:-1], label='{}={}'.format(param_k, k_med))
    ax1.plot(k_high[0][:-1], label='{}={}'.format(param_k, high_k))
    ax1.legend(loc="upper right")

    ax2.set(ylabel='Inventory', xlabel='time (minutes)')
    ax2.plot(k_low[1], label='{}={}'.format(param_k, low_k))
    # ax2.plot(kmu[1], label='{}={}'.format(param_k, k_med))
    ax2.plot(k_high[1], label='{}={}'.format(param_k, high_k))
    ax2.legend(loc="upper right")

    ax3.set(ylabel='Income', xlabel='time (minutes)')
    ax3.plot(b_low[0][:-1], label='{}={}'.format(param_b, low_b))
    # ax3.plot(bmu[0][:-1], label='{}={}'.format(param_b, b_med))
    ax3.plot(b_high[0][:-1], label='{}={}'.format(param_b, high_b))
    ax3.legend(loc="upper right")

    ax4.set(ylabel='Inventory', xlabel='time (minutes)')
    ax4.plot(b_low[1], label='{}={}'.format(param_b, low_b))
    # ax4.plot(bmu[1], label='{}={}'.format(param_b, b_med))
    ax4.plot(b_high[1], label='{}={}'.format(param_b, high_b))
    ax4.legend(loc="upper right")
    plt.tight_layout()
    plt.show()




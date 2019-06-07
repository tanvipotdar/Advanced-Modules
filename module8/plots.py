import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from creditgrades import creditgrades


R = 0.4
L_mean = 0.5
S_ref = 100
lmbda = 0.3
r = 0.05

# plot of cds spread vs vol
def plot_cds_spread_term_structure_vs_vol():
    t = pd.Series(list(range(1,11)))
    S0 = pd.Series([100]*10)
    D = pd.Series([50]*10)
    df_30 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=pd.Series([.30]*10), D=D, lmbda=lmbda, r=r, R=R, t=t)
    df_45 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=pd.Series([.45]*10), D=D, lmbda=lmbda, r=r, R=R, t=t)
    df_60 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=pd.Series([.60]*10), D=D, lmbda=lmbda, r=r, R=R, t=t)
    df_75 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=pd.Series([.75]*10), D=D, lmbda=lmbda, r=r, R=R, t=t)
    df_cds = pd.DataFrame()
    df_cds['vol=.30'] = df_30['cds_spread']
    df_cds['vol=.45'] = df_45['cds_spread']
    df_cds['vol=.60'] = df_60['cds_spread']
    df_cds['vol=.75'] = df_75['cds_spread']
    df_cds.index = df_30.maturity

    df_sp = pd.DataFrame()
    df_sp['vol=.30'] = 1-df_30['survival_probability']
    df_sp['vol=.45'] = 1-df_45['survival_probability']
    df_sp['vol=.60'] = 1-df_60['survival_probability']
    df_sp['vol=.75'] = 1-df_75['survival_probability']
    df_sp.index = df_30.maturity

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    ax1.plot(df_cds)
    ax2.plot(df_sp)
    ax1.set_ylabel('CDS Spread (bps)')
    ax1.set_xlabel('Maturity (years)')
    ax2.set_ylabel('Default Probability')
    ax2.set_xlabel('Maturity (years)')
    plt.tight_layout()
    plt.show()


# plot of cds spread vs s/d
def plot_cds_spread_term_structure_vs_sdratio():
    t = pd.Series(list(range(1,11)))
    S0 = pd.Series([100]*10)
    D = pd.Series(100/np.linspace(0.5,5,10))
    sigma_ref=pd.Series([.30]*10)
    df_30 = creditgrades(L_mean=L_mean, S0=pd.Series([100]*10), S_ref=S_ref, sigma_ref=sigma_ref, D=pd.Series([200]*10), lmbda=lmbda, r=r, R=R, t=t)
    df_45 = creditgrades(L_mean=L_mean, S0=pd.Series([100]*10), S_ref=S_ref, sigma_ref=sigma_ref, D=pd.Series([66.6]*10), lmbda=lmbda, r=r, R=R, t=t)
    df_60 = creditgrades(L_mean=L_mean, S0=pd.Series([100]*10), S_ref=S_ref, sigma_ref=sigma_ref, D=pd.Series([40]*10), lmbda=lmbda, r=r, R=R, t=t)
    df_75 = creditgrades(L_mean=L_mean, S0=pd.Series([100]*10), S_ref=S_ref, sigma_ref=sigma_ref, D=pd.Series([20]*10), lmbda=lmbda, r=r, R=R, t=t)
    df_cds = pd.DataFrame()
    df_cds['S0/D=0.5'] = df_30['cds_spread']
    df_cds['S0/D=1.5'] = df_45['cds_spread']
    df_cds['S0/D=2.5'] = df_60['cds_spread']
    df_cds['S0/D=5.0'] = df_75['cds_spread']
    df_cds.index = df_30.maturity
    ax = df_cds.plot()
    ax.set_ylabel('CDS Spread (bps)')
    ax.set_xlabel('Maturity (years)')
    plt.show()


# plot of

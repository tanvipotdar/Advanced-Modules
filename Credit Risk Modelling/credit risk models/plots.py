import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from creditgrades import creditgrades


R = 0.4
L_mean = 0.5
S_ref = 100
lmbda = 0.3
r = 0.05


def plot_cds_and_pd(df_cds, df_sp):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,3))
    df_cds.plot(ax=ax1)
    df_sp.plot(ax=ax2)
    ax1.set_ylabel('CDS Spread (bps)')
    ax1.set_xlabel('Maturity (years)')
    ax1.legend()
    ax2.set_ylabel('Default Probability (%)')
    ax2.set_xlabel('Maturity (years)')
    ax2.legend()
    plt.tight_layout()
    plt.show()


# plot of cds spread vs vol
def plot_cds_spread_term_structure_vs_vol():
    n = 20
    t = pd.Series(list(range(1,21)))
    S0 = pd.Series([100]*n)
    D = pd.Series([100]*n)
    df_30 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=pd.Series([.20]*n), D=D, lmbda=lmbda, r=r, R=R, t=t)
    df_45 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=pd.Series([.40]*n), D=D, lmbda=lmbda, r=r, R=R, t=t)
    df_60 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=pd.Series([.60]*n), D=D, lmbda=lmbda, r=r, R=R, t=t)
    df_75 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=pd.Series([.80]*n), D=D, lmbda=lmbda, r=r, R=R, t=t)
    df_cds = pd.DataFrame()
    df_cds['vol=20%'] = df_30['cds_spread']
    df_cds['vol=40%'] = df_45['cds_spread']
    df_cds['vol=60%'] = df_60['cds_spread']
    df_cds['vol=80%'] = df_75['cds_spread']
    df_cds.index = df_30.maturity
    df_sp = pd.DataFrame()
    df_sp['vol=20%'] = (1-df_30['survival_probability'])*100
    df_sp['vol=40%'] = (1-df_45['survival_probability'])*100
    df_sp['vol=60%'] = (1-df_60['survival_probability'])*100
    df_sp['vol=80%'] = (1-df_75['survival_probability'])*100
    df_sp.index = df_30.maturity
    plot_cds_and_pd(df_cds, df_sp)


# plot of cds spread vs L
def plot_cds_spread_term_structure_vs_leverage():
    n = 20
    t = pd.Series(list(range(1,21)))
    S0 = pd.Series([100]*n)
    sigma_ref=pd.Series([.80]*n)
    # df_20 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=pd.Series([100/0.25]*n), lmbda=lmbda, r=r, R=R, t=t)
    df_30 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=pd.Series([100/0.5]*n), lmbda=lmbda, r=r, R=R, t=t)
    df_45 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=pd.Series([100/1.5]*n), lmbda=lmbda, r=r, R=R, t=t)
    df_60 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=pd.Series([100/2.5]*n), lmbda=lmbda, r=r, R=R, t=t)
    df_75 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=pd.Series([100/5.0]*n), lmbda=lmbda, r=r, R=R, t=t)
    df_cds = pd.DataFrame()
    # df_cds['L=0.25'] = df_20['cds_spread']
    df_cds['S/D=0.5'] = df_30['cds_spread']
    df_cds['S/D=1.5'] = df_45['cds_spread']
    df_cds['S/D=2.5'] = df_60['cds_spread']
    df_cds['S/D=5'] = df_75['cds_spread']
    df_cds.index = df_30.maturity
    df_sp = pd.DataFrame()
    # df_sp['L=0.25'] = (1-df_20['survival_probability'])*100
    df_sp['S/D=0.5'] = (1-df_30['survival_probability'])*100
    df_sp['S/D=1.5'] = (1-df_45['survival_probability'])*100
    df_sp['S/D=2.5'] = (1-df_60['survival_probability'])*100
    df_sp['S/D=5'] = (1-df_75['survival_probability'])*100
    df_sp.index = df_30.maturity
    plot_cds_and_pd(df_cds, df_sp)


# plot of cds spread vs lambda
def plot_cds_spread_term_structure_vs_lambda():
    n = 20
    t = pd.Series(list(range(1,21)))
    S0 = pd.Series([100]*n)
    D = pd.Series([100]*n)
    sigma_ref=pd.Series([.25]*n)
    df_30 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=D, lmbda=0.05, r=r, R=R, t=t)
    df_60 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=D, lmbda=0.2, r=r, R=R, t=t)
    df_75 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=D, lmbda=0.4, r=r, R=R, t=t)
    df_76 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=D, lmbda=0.6, r=r, R=R, t=t)
    df_cds = pd.DataFrame()
    df_cds['lambda=0.05'] = df_30['cds_spread']
    df_cds['lambda=0.2'] = df_60['cds_spread']
    df_cds['lambda=0.4'] = df_75['cds_spread']
    df_cds['lambda=0.6'] = df_76['cds_spread']
    df_cds.index = df_30.maturity
    fig = df_cds.plot(figsize=(6,3))
    fig.set_xlabel('Maturity (years)')
    fig.set_ylabel('CDS Spread (bps)')
    plt.tight_layout()
    plt.show()



# plot of cds spread vs lambda
def plot_cds_spread_term_structure_vs_recovery_rate():
    n = 20
    t = pd.Series(list(range(1,21)))
    S0 = pd.Series([100]*n)
    D = pd.Series([100]*n)
    sigma_ref=pd.Series([.25]*n)
    df_30 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=D, lmbda=lmbda, r=r, R=0.1, t=t)
    df_60 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=D, lmbda=lmbda, r=r, R=0.3, t=t)
    df_75 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=D, lmbda=lmbda, r=r, R=0.6, t=t)
    df_76 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=D, lmbda=lmbda, r=r, R=0.9, t=t)
    df_R = pd.DataFrame()
    df_R['R=0.1'] = df_30['cds_spread']
    df_R['R=0.3'] = df_60['cds_spread']
    df_R['R=0.6'] = df_75['cds_spread']
    df_R['R=0.9'] = df_76['cds_spread']
    df_R.index = df_30.maturity

    df_30 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=D, lmbda=lmbda, r=0.03, R=R, t=t)
    df_60 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=D, lmbda=lmbda, r=0.08, R=R, t=t)
    df_75 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=D, lmbda=lmbda, r=0.3, R=R, t=t)
    df_76 = creditgrades(L_mean=L_mean, S0=S0, S_ref=S_ref, sigma_ref=sigma_ref, D=D, lmbda=lmbda, r=0.8, R=R, t=t)
    df_r = pd.DataFrame()
    df_r['r=0.03'] = df_30['cds_spread']
    df_r['r=0.08'] = df_60['cds_spread']
    df_r['r=0.3'] = df_75['cds_spread']
    df_r['r=0.8'] = df_76['cds_spread']
    df_r.index = df_30.maturity
    plot_cds_and_pd(df_R, df_r)





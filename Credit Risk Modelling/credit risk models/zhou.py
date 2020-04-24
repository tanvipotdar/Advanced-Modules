import numpy as np


M=10000
T=1
n=100
r=0.05 #risk-free interest rate
l=0.3 #intensity parameter of poisson process
nu=2 #expected value of jump amplitude
sigma_v=0.35 #asset vol excluding jumps
mu_v=0 #asset drift
sigma_pi=0.1 #vol of jump size
mu_pi=0.1 #drift of jump size
V0=100 #initial asset value
K=100 #strike price
def_time = 0;

X=np.zeros(V)

for j in range(M):
    for i in range(n):
        mu_x = (r-0.5*sigma_v**2-l*nu)*(T/i)
        sigma_x = sigma_v*np.sqrt((T/i))
        x_i = np.random.normal(mu,sigma,1)
        pi_i = np.random.normal(mu_pi,sigma_pi,1)
        y_i=0 if 1-l*(T/i) else 1
        X[i]=X[i-1]+x_i+y_i*pi_i if i>0 else np.log(V[0])
        if X[i] <= np.log(K):
            def_time = i/T
    

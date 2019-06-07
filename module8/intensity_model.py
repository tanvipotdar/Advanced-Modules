import numpy as np
from scipy import integrate, optimize

r = 0.03
t = 0
mat = 1
R = 0.0192
L = 0.6


def bond_pricer(t, r, T):
    return np.exp(-r*(T-t)/T)

def cds(gamma):
	'''
	t - cds is priced as of this time
	mat - maturity of the cds
	R - cds spread
	L - loss given default
	gamma - intensities for each maturity year per quarter
	P - price of risk-free zero-coupon bond
	'''
	q = (mat-t)*4
	gamma = [gamma]*q
	Gamma = np.cumsum(gamma)

	a1 = 0
	a2 = 0
	a3 = 0

	for i in range(1, q):
		f1 = lambda u : np.exp(-Gamma[i-1] - gamma[i]*(u-(i-1)))*bond_pricer(u,r,q)*(u-(i-1))
		a1 += gamma[i] + integrate.quad(f1, i-1, i)[0]

		a2 += bond_pricer(i,r,q)*np.exp(-Gamma[i])

		f3 = lambda u : np.exp(-Gamma[i-1] - gamma[i]*(u-(i-1)))*bond_pricer(u,r,q)
		a3 += gamma[i] + integrate.quad(f1, i-1, i)[0]

	a1 = a1*R
	a2 = a2*R
	a3 = L*a3

	return a1 + a2 -a3

x = optimize.fsolve(cds,0.0125)


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
    

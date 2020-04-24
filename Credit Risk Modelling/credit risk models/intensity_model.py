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


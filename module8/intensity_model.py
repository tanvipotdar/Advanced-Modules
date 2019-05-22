import numpy as np
from scipy import integrate


def cds(t, mat, R, L, gamma, P):
	'''
	t - cds is priced as of this time
	mat - maturity of the cds
	R - cds spread
	L - loss given default 
	gamma - intensities for each maturity year 
	P - price of risk-free zero-coupon bond
	'''
	quarters_left = (mat-t)*4
	Gamma = np.zeros(quarters_left)
	for i in range(1, len(Gamma)):
		Gamma[i] = Gamma[i-1] + gamma[i]

	a1 = 0
	a2 = 0
	a3 = 0

	for i in range(len(Gamma)):
		a1 += gamma[i] + integrate.quad(np.exp(-Gamma[i] - gamma[i]*))
		a2 += np.exp(-Gamma[i])

		f3 = lambda i,u: np.exp(-Gamma[i-1] - gamma[i](u - (i-1))) * P(u*90)
		a3 += gamma[i] * integrate.quad()
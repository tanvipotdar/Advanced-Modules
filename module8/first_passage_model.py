'''
Scenario Barrier Time-Varying Vol AT1P model
'''

import pandas as pd
import numpy as np
from scipy.stats import norm


def get_cds_data():
	cds_data = pd.DataFrame(columns=['maturity', 'spread', 'rate'],
		data = [[0.5, -0.0028, 0.0063], [1.0, -0.0024, 0.0073], [2.0, -0.0017, 0.0091], 
				[3.0, -0.008, 0.0110], [4.0, 0.002, 0.0136], [5.0, 0.0014, 0.016], [7.0, 0.0039, 0.0183],
				[10.0, 0.0076, 0.0199], [20.0, 0.0137, 0.0207], [30.0, 0.0146, 0.0209]])
	return cds_data


def at1p(V0, H0, B, sigma, r, t):
	dt = np.diff([0]+t.tolist())
	sigma2u = np.cumsum(sigma**2*dt)
	a1 = norm.cdf((np.log(V0/H0) + 0.5 *(2*B-1)*sigma2u)/np.sqrt(sigma2u))
	a2 = (H0/V0)**(2*B-1)
	a3 = norm.cdf((np.log(H0/V0) + 0.5 *(2*B-1)*sigma2u)/np.sqrt(sigma2u))
	survival_prob = a1-(a1*a3)

	Vt = V0*np.exp(r*t)

	Ht = np.zeros(len(t))
	for i in range(1,len(t)):
		Ht[i] = (H0/V0) * Vt[i] * np.exp(-B*sigma2u[i])

	intensity = np.zeros(len(t))
	intensity[1] = np.log(survival_prob[1])/dt[1]
	for i in range(2, len(t)):
		intensity[i] = (-np.log(survival_prob[i])+np.log(survival_prob[i-1])) / dt[i]

	result = pd.DataFrame(data=[t,Vt,Ht,survival_prob, intensity]).T
	result.columns = ['Maturity', 'Firm Value', 'H', 'Survivial Probability', 'Default Intensity']
	return result


cds_data = get_cds_data()
df = at1p(V0=1, H0=0.7,B=0.4,sigma=np.repeat(0.1, 10), r=cds_data.rate, t=cds_data.maturity)


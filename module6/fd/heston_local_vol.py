# Calculates the local volatility in the Heston model
import numpy as np
from transition_density import calculate_transition_density_and_vol
from scipy.sparse import spdiags
from scipy.interpolate import interp2d


def compute_option_price(I=4, M=4):
	T=3;
	r=0.03;

	h=Smax/N;
	S=np.linspace(0, Smax, I+1) 

	dt=T/M;
	theta=0.;

	V=max(K-S,0);

	_, vol =. calculate_transition_density_and_vol()

	for m in range(M, -1, -1):
		sigma = vol[m]
		term1=sigma*S/h;
		term1=term1.*term1;
		term2=r*S/h;

		A=0.5*dt*(term1-term2);
		B=1-dt*(term1+r);
		C=0.5*dt*(term1+term2);

		Mat=spdiags([C B A],-1:1,I+1,I+1)
	    V=Mat*V;

    f = interp2d(S, V.reshape(I, J))
    return f(S0, Y0)


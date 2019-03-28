import numpy  as np
from numpy.random import randn
from numpy import ones, zeros, eye, sqrt, exp, log2, diag, reshape
from numpy.linalg import cholesky, eig
from scipy.stats import norm
from bb import bb
from sobol import sobol, scramble


# param values
K = 20.
S0 = 20.
T = 0.5
r = 0.01
sigma = 0.2
rho = 0.1

for option in range(1,4):
    print('---------------------------')
    if   option == 1:
        print('Basket call option\n')
    elif option == 2:
        print('Basket Asian strike call option\n')
    elif option == 3:
        print('Basket lookback put option\n')
    print('---------------------------')

    # two passes for Cholesky and PCA factorisations of correlation matrix

    for opass in [1,2]:
    	sigma_matrix = eye(4) + rho*(ones((4,4))-eye(4))

        if opass == 1:
            print('Cholesky factorisation of correlation:\n')
            L = cholesky(sigma_matrix)
        else:
            print('PCA factorisation of correlation:\n')
            D,V = eig(sigma_matrix)
            L = V.dot(diag(sqrt(D)))

        # three inner passes for Cholesky and Brownian Bridge factorisations
        # of covariance matrix in time, and use of plain MC

        for ipass in range(1,4):
            N  = 64            # number of timesteps
            M2 = 64            # number of randomisations
            M  = int(2**18/M2) # number of paths in each "family"

            unscrambled = sobol(m=int(log2(M)), s=4*N, scramble=False)

            h  = T/N

            sum1 = 0.
            sum2 = 0.

            if ipass == 1:
                print('Sobol with BB')
            elif ipass == 2:
                print('Sobol without BB')
            else:
                print('plain MC')

            for m in range(1,M2+1):
                if ipass == 1:
                    # Sobol points with Brownian Bridge construction of Brownian increments
                    U  = scramble(unscrambled).T  # generate set of M Sobol points
                    Z  = norm.ppf(U)         # inverts Normal cum. fn.
                    dW = bb(Z,T,L)

                elif ipass == 2:
                    # Sobol points without Brownian Bridge construction                    
                    U  = scramble(unscrambled).T  # generate set of M Sobol points
                    Z  = norm.ppf(U)        
                    Z  = reshape(Z, (4,N*M), order='F')
                    dW = sqrt(h)*L.dot(Z)
                    dW = reshape(dW, (4*N,M), order='F')

                else:
                    # standard random number generation
                    dW = sqrt(h)*L.dot(randn(4,N*M))
                    dW = reshape(dW, (4*N,M), order='F')

                S    = S0*ones((4,M))
                Save = zeros((M,))
                Smin = S0*ones((M,))

                for n in range(1,N+1):
                    S    = S*(1+r*h+sigma*dW[4*n-4:4*n,:])
                    Save = Save + 0.25*np.sum(S,0)/N
                    Smin = np.minimum(Smin,0.25*np.sum(S,0))

                S = 0.25*np.sum(S,0)
                if   option == 1:
                    P = exp(-r*T)*np.maximum(S-K,0)
                elif option == 2:
                    P = exp(-r*T)*np.maximum(S-Save,0)
                else:
                	P = exp(-r*T)*np.maximum(K-Smin,0)

                P = np.sum(P)/M

                sum1 = sum1 + np.sum(P)
                sum2 = sum2 + np.sum(P**2)

            V  = sum1/M2
            sd = sqrt((sum2/M2 - V**2)/(M2-1))
            if ipass == 3:
                print(' MC_val      = %f \n' % V)
                print(' MC_std_dev  = %f \n\n' % sd)
            else:
                print(' QMC_val      = %f \n' % V)
                print(' QMC_std_dev  = %f \n\n' % sd)           



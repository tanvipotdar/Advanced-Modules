
import numpy as np

# define all parameters
T = 1                           # expiry (hours)
N = 1e4                         # number of shares
mc_paths = 10000                # number of monte carlo simulations
s_sigma = 0.027                 # vol of the stock
s0 = 100                        # initial stock price
phi = 1e-3                      # running inventory penalty
alpha = 1e-3                    # terminal liquidation penalty
k_eta = 0.1                     # mean reversion rate of k
k_mu = 1e-4                     # long running mean of k
k_sigma = 1.5e-3                # vol of k

b_eta = 0.01                    # mean reversion rate of b
b_mu = 1e-4                     # long running mean of b
b_sigma = 3e-3                  # vol of b


def run_almgren_chriss_with_constant_pi():
    # time in minutes
    t = T*60

    # stochastic processes
    w = np.random.randn(t+1)	# brownian motion for stock price, normal random variables
    s = np.zeros(t+1)			# mid-price process
    e = np.zeros(t+1)			# execution price process
    v = np.zeros(t+1)			# trading speed process
    x = np.zeros(t+1)			# income process
    q = np.zeros(t+1)			# shares to liquidate
    q_liq = np.zeros(t+1)		# shares liquidated 


    q[0] = N
    gamma = np.sqrt(phi/k_mu)
    zeta = (alpha - 0.5*b_mu + np.sqrt(phi*k_mu))/(alpha - 0.5*b_mu - np.sqrt(phi*k_mu))

    for i in range(t+1):
    	s[i] = s0 if i==0 else q[i-1]- b_mu*v[i-1]/t + s_sigma*w[i]*np.sqrt(t)
    	tT = (t-i)/t
    	v_num = zeta*np.exp(gamma*tT) + np.exp(-gamma*tT)
    	v_denom = zeta*np.exp(gamma*T) + np.exp(-gamma*T)
    	v[i] = gamma*(v_num/v_denom) * q[i]
    	v[i] = max(v[i],0)/60
    	e[i] = s[i] - k_mu*v[i]
    	q_liq[i] = min(v[i],q[i])
    	x[i] = e[i]*q_liq[i]
    	if i==t:
    		x[i] = 0
    		q[i] = q[i-1]
    	else:
    		q[i+1] = q[i] - q_liq[i]
    		x[i] = e[i]*q_liq[i]

    x[t] = q[t]*(s[t] - alpha*q[t])
    terminal_x = np.sum(x)
    return terminal_x


def calculate_stochastic_path(mu, eta, sigma):
	t = T*60					
	w = np.random.randn(t+1)		# brownian motion for the price impact param
	pi = np.zeros(t+1)				# path for the price impact param

	for i in range(t+1):
		tT = (t-i)/t
		pi[i] = mu + sigma*np.sqrt((np.exp(-2*eta*tT))/(2*eta))*w[i]
	return pi



def run_almgren_chriss_with_stochastic_pi():
    # time in minutes
    t = T*60

    # stochastic processes
    w = np.random.randn(t+1)	# brownian motion for stock price, normal random variables
    s = np.zeros(t+1)			# mid-price process
    e = np.zeros(t+1)			# execution price process
    v = np.zeros(t+1)			# trading speed process
    x = np.zeros(t+1)			# income process
    q = np.zeros(t+1)			# shares to liquidate
    q_liq = np.zeros(t+1)		# shares liquidated 

    # coefficients for optimal speed
    gamma = np.zeros(t+1)		
    zeta = np.zeros(t+1)

    q[0] = N
    k = calculate_stochastic_path(k_mu, k_eta, k_sigma)
    b = calculate_stochastic_path(b_mu, b_eta, b_sigma)

    for i in range(t+1):
    	gamma[i] = np.sqrt(phi/k[i])
    	zeta[i] = (alpha - 0.5*b[i] + np.sqrt(phi*k[i]))/(alpha - 0.5*b[i] - np.sqrt(phi*k[i]))

    	s[i] = s0 if i==0 else q[i-1]- b[i]*v[i-1]/t + s_sigma*w[i]*np.sqrt(t)
    	v_num = zeta[i]*np.exp(gamma[i]*tT) + np.exp(-gamma[i]*tT)
    	v_denom = zeta[i]*np.exp(gamma[i]*T) + np.exp(-gamma[i]*T)
    	v[i] = gamma[i]*(v_num/v_denom) * q[i]
    	v[i] = max(v[i],0)/60
    	e[i] = s[i] - k_mu*v[i]
    	q_liq[i] = min(v[i],q[i])

    	if i==t:
    		x[i] = 0
    		q[i] = q[i-1]
    	else:
    		q[i+1] = q[i] - q_liq[i]
    		x[i] = e[i]*q_liq[i]

    x[t] = q[t]*(s[t] - alpha*q[t])
    terminal_x = np.sum(x)
    return terminal_x


def calculate_performance():
	cash_from_constant_strategy = np.zeros(mc_paths+1)
	cash_from_stochastic_strategy = np.zeros(mc_paths+1)
	performance = np.zeros(mc_paths+1)

	for x in range(mc_paths):
		cash_from_constant_strategy[x] = run_almgren_chriss_with_constant_pi()
		cash_from_stochastic_strategy[x] = run_almgren_chriss_with_stochastic_pi()
		performance[x] = (cash_from_stochastic_strategy[x] - cash_from_constant_strategy[x])/cash_from_constant_strategy[x] * mc_paths

	return performance

	













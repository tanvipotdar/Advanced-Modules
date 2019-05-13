# default - when asset value crosses a predetermined barrier or threshold
# V - asset value process for the firm on a per share basis/ gbm
# L - average recovery on debt/ lognormal random var
# D - firm's debt per share

# constants
v0 = 100            # initial asset value
v_sigma = 0.01      # asset vol
v_mu = 0            # asset drift

l = 0.5             # mean of recovery rate
l_sigma = 0.09      # vol of recovery rate

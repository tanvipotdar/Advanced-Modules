import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#helper functions
 def f(x):            
	if x.year>2018:
		x = x.replace(year=x.year-100)
	return x


gnp_data = pd.read_csv('/Users/tanvi/Projects/modelling/GNP.csv', index_col='DATE')
log_returns = np.log(gnp_data.GNP/gnp_data.shift(1).GNP)*100
log_returns.fillna(1.0, inplace=True)
dates_index = pd.to_datetime(log_returns.index)
dates = map(f, dates_index)
log_returns.index = dates

training_data = log_returns[:232]
testing_data = log_returns[232:]

# plotting training data log returns
training_returns_plot = training_data.plot()
training_returns_plot.set(xlabel="Time(Quarter)", ylabel="GNP quarterly growth rate", title="Log returns over time")

#1.2 Exponential Smoothing (b)
ses = training_data.to_frame('y')
alpha_values = np.arange(0.01, 1.01, 0.01)

for alpha in alpha_values:
	col_name = 's_{}'.format(alpha)
	#TODO: Check if this should be 1.0
	ses[col_name]=ses.y.tolist()[0]
	for i in range(1, len(ses)):
		ses[col_name][i] = ses[col_name][i-1]*(1-alpha) + alpha*ses.y[i]

errors = pd.DataFrame()
errors['alpha'] = alpha_values
error_vals = []
for alpha in alpha_values:    
	error_diff = (ses['s_{}'.format(alpha)].shift(1)-ses['y']).dropna()                  
	error_val = sum(error_diff**2)
	error_vals.append(error_val)

 errors['sse'] = error_vals
 errors.set_index('alpha', inplace=True)
 optimal_alpha = errors.idxmin()
 sse_plot = errors.plot()
 sse_plot.set(xlabel="Alpha", ylabel="Sum of squared errors", title="SSE for alpha values")
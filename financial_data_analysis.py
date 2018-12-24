import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARMA

#helper functions
 def f(x):            
	if x.year>2018:
		x = x.replace(year=x.year-100)
	return x

# 1.1
gnp_data = pd.read_csv('/Users/tanvipotdar/Projects/Advanced-Numerical-Methods/GNP.csv', index_col='DATE')
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

# plotting training data acf
acf_plot = plot_acf(training_data)
acf_plot.set(xlabel="Lags", ylabel="Autocorrelation", title="Autocorrelation function for log returns")

# plotting training data pacf
pacf_plot = plot_pacf(training_data)
pacf_plot.set(xlabel="Lags", ylabel="Partial Autocorrelation", title="Partial Autocorrelation function for log returns")

# 1.2 Exponential Smoothing 
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

# plot of alpha values versus in sample SSE
sse_plot = errors.plot()
sse_plot.set(xlabel="Alpha", ylabel="Sum of squared errors", title="SSE for alpha values")

# optimal value of alpha that minimises in sample SSE
optimal_alpha = errors.idxmin()

# plot of in-sample model response vs optimal alpha 
optimal_df = ses[['y', 's_0.52']]
optimal_df = optimal_df.rename(columns={'y':'Actual GNP Returns', 's_0.52':'GNP Returns using optimal alpha'})
optimal_plot = optimal_df.plot()
optimal_plot.set(xlabel="Time(Quarter)", ylabel="SES Model GNP Returns", title="Optimal SES vs Actual GNP Returns")

# 1.3 Autoregressive models

# Check why AR and ARMA of same order return different results
# ar_model = AR(training_data)
# min_aic = 20000000
# min_order=0
# for p in range(1,21):
# 	model_fit = ar_model.fit(p)
# 	model_aic = model_fit.aic
# 	if model_aic < min_aic:
# 		min_aic = model_aic
# 		min_order = p


min_aic = 20000000
min_order=0
for p in range(1,21):
	model = ARMA(training_data, order=(p,0))
	model_fit = model.fit()
	model_aic = model_fit.aic
	if model_aic < min_aic:
		min_aic = model_aic
		min_order = p


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
log_returns.dropna(inplace=True)
dates_index = pd.to_datetime(log_returns.index)
dates = map(f, dates_index)
log_returns.index = dates

training_data = log_returns[:231]
testing_data = log_returns[231:]


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
	model = ARMA(training_data[1:], order=(p,0))
	model_fit = model.fit()
	model_aic = model_fit.aic
	if model_aic < min_aic:
		min_aic = model_aic
		min_order = p

# Out of sample forecasting

persistence_forecast = []
climatology_forecast = []
ses_forecast = []
ar_forecast = []

avg_vals = [training_data[:-k].mean() if k>0 else training_data.mean() for k in range(12,-1,-1)]

ar_optimal = ARMA(training_data, order=(16,0))
ar_fit = ar_optimal.fit()
params = ar_fit.params
residuals = ar_fit.resid
p = ar_fit.k_ar
q = ar_fit.k_ma
k_exog = ar_fit.k_exog
k_trend = ar_fit.k_trend
for k in range(1,13):
	persistence_forecast.append(pd.Series(data=training_data[-k:].values, index=testing_data[:k].index))
	climatology_forecast.append(pd.Series(data = avg_vals[-k:], index=testing_data.index[:k]))
	ses_forecast.append(pd.Series(data=optimal_df['GNP Returns using optimal alpha'][-k:].values, index=testing_data[:k].index))
	ar_prediction = pd.Series(data=_arma_predict_out_of_sample(params, k, residuals, p, q, k_trend, k_exog, endog=training_data[1:], exog=None, start=len(training_data[1:])), index=testing_data[:k].index)
	ar_forecast.append(ar_prediction)

ar_prediction = pd.Series(data=_arma_predict_out_of_sample(params, len(testing_data), residuals, p, q, k_trend, k_exog, endog=training_data, exog=None, start=len(training_data)), index=testing_data.index)
preds=ar_prediction.to_frame('Optimal AR(16) predictions')
td =  testing_data.to_frame('Actual GNP Returns')
ar_vs_td = td.join(preds)

ar_vs_td_plot = ar_vs_td.plot()
ar_vs_td_plot.set(xlabel="Time(Quarter)", ylabel="AR Model GNP Returns", title="Optimal AR vs Actual GNP Returns")

pdf_plot = ar_vs_td[['Out-of-sample errors']].plot.kde()
pdf_plot.set(title="Probability Density Function of out of sample errors")

rmse_persistence_data = []
mae_per_data = []
for p in persistence_forecast:
	k = len(p)
	e = p - testing_data.head(k)
	rmse_persistence_data.append((e**2).mean()**.5)
	mae_per_data.append(e.abs().mean())
rmse_persistence = pd.Series(data=rmse_persistence_data, index=range(1,13)).to_frame('Persistence')
mae_persistence = pd.Series(data=mae_per_data, index=range(1,13)).to_frame('Persistence')

rmse_climatology_data = []
mae_clim_data = []
for p in climatology_forecast:
	k = len(p)
	e = p - testing_data.head(k)
	rmse_climatology_data.append((e**2).mean()**.5)
	mae_clim_data.append(e.abs().mean())
rmse_climatology = pd.Series(data=rmse_climatology_data, index=range(1,13)).to_frame('Climatology')
mae_climatology = pd.Series(data=mae_clim_data, index=range(1,13)).to_frame('Climatology')

rmse_ses_data = []
ses_mae_data = []
for p in ses_forecast:
	k = len(p)
	e = p - testing_data.head(k)
	rmse_ses_data.append((e**2).mean()**.5)
	ses_mae_data.append(e.abs().mean())
rmse_ses = pd.Series(data=rmse_ses_data, index=range(1,13)).to_frame('SES Optimal')
mae_ses = pd.Series(data=ses_mae_data, index=range(1,13)).to_frame('SES Optimal')


rmse_ar_data = []
mae_ar_data = []
for p in ar_forecast:
	k = len(p)
	e = p - testing_data.head(k)
	rmse_ar_data.append((e**2).mean()**.5)
	mae_ar_data.append(e.abs().mean())
rmse_ar = pd.Series(data=rmse_ar_data, index=range(1,13)).to_frame('AR Optimal')
mae_ar = pd.Series(data=mae_ar_data, index=range(1,13)).to_frame('AR Optimal')


rmse_df = rmse_persistence.join(rmse_climatology).join(rmse_ses).join(rmse_ar)
rmse_plot = rmse_df.plot()
rmse_plot.set(xlabel="Horizon", ylabel="RMSE", title="RMSE")


mae_df = mae_persistence.join(mae_climatology).join(mae_ses).join(mae_ar)
mae_plot = mae_df.plot()
mae_plot.set(xlabel="Horizon", ylabel="MAE", title="MAE")

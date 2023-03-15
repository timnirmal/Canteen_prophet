import pandas as pd
from matplotlib import pyplot
from prophet import Prophet
from lib.datetimegen import generate_datetimes as dgt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from prophet.serialize import model_to_json, model_from_json
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import numpy as np
from dask.distributed import Client

# client = Client(n_workers=4, threads_per_worker=1, memory_limit='2GB')
client = Client()

# load every column in pd head
pd.set_option('display.max_columns', None)

# load the data/transposed first column
df = pd.read_csv("data/df3.csv")

# keep only the datetime and sales columns
df = df[["ds", "y"]]
# datetime column to datetime type
df["ds"] = pd.to_datetime(df["ds"])

print(df.dtypes)
print(df.head())

# load holidays
holidays = pd.read_csv("data/holidays-2.csv")

playoffs = pd.DataFrame({
    'holiday': 'playoff',
    # holidays as 'ds'
    'ds': holidays["ds"],
    'lower_window': 0,
    'upper_window': 1,
})

holidays = playoffs

# # add holidays
# model = Prophet(
#     holidays=holidays,
#     changepoint_prior_scale=1, changepoint_range=1,
#     yearly_seasonality=True, weekly_seasonality=True,
#     daily_seasonality=True,
#     seasonality_mode='multiplicative',
#     seasonality_prior_scale=500.0, holidays_prior_scale=500.0, mcmc_samples=0, interval_width=1,
#     uncertainty_samples=2000, stan_backend=None,
#     # growth='logistic'
# )

# add regressors
# model.add_regressor("y")

# Metrics
# R2:  0.383919577279893
# MSE:  2427.8544877382374
# MAE:  17.901745566393274

# R2:  0.4085147857617024
# MSE:  2330.9295002085705
# MAE:  18.468722578249313

cutoffs = pd.to_datetime(['2020-06-01', '2021-06-01', '2022-06-01'])

param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5, 1],
    'seasonality_prior_scale': [0.01, 0.1, 1],
    'holidays': [holidays],
    'seasonality_mode': ['multiplicative'],
    'holidays_prior_scale': [10],
    'mcmc_samples': [0],
    'interval_width': [1],
    'uncertainty_samples': [2000],
    'stan_backend': [None],
    'changepoint_range': [1],
    'yearly_seasonality': [True],
    'weekly_seasonality': [True],
    'daily_seasonality': [True],
}

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    m = Prophet(**params).fit(df)  # Fit model with given params
    # df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')
    df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days', parallel="dask")
    df_p = performance_metrics(df_cv)
    rmses.append(df_p['rmse'].values[0])

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)

best_params = all_params[np.argmin(rmses)]
print('Best params: ', best_params)

#
# # fit the model
# model.fit(df)
#
# future = dgt(1200, "2020-01-01")
# # save to csv
# future.to_csv("data/future_gen.csv", index=False)
#
# future = model.make_future_dataframe(periods=200, freq='D')
#
# future.to_csv("data/future_gen_prophet.csv", index=False)
#
# print("future")
# print(future.head())
#
# # use the model to make a forecast
# forecast = model.predict(future)
#
# # diagnostics
# df_cv = cross_validation(model, initial='730 days', period='180 days', horizon = '365 days')
# df_p = performance_metrics(df_cv)
#
# print("df_cv")
# print(df_cv.head())
# print("df_p")
# print(df_p.head())
#
#
# # summarize the forecast
# print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
#
# r2_score(df["y"], forecast["yhat"][:len(df["y"])])
# mean_squared_error(df["y"], forecast["yhat"][:len(df["y"])])
# mean_absolute_error(df["y"], forecast["yhat"][:len(df["y"])])
#
# print("Metrics")
# print("R2: ", r2_score(df["y"], forecast["yhat"][:len(df["y"])]))
# print("MSE: ", mean_squared_error(df["y"], forecast["yhat"][:len(df["y"])]))
# print("MAE: ", mean_absolute_error(df["y"], forecast["yhat"][:len(df["y"])]))
#
#
#
# # plot forecast
# model.plot(forecast)
# pyplot.show()
#
# # plot forecast components
# model.plot_components(forecast)
# pyplot.show()
#
# from prophet.plot import add_changepoints_to_plot
#
# fig = model.plot(forecast)
# a = add_changepoints_to_plot(fig.gca(), model, forecast)
# pyplot.show()


# with open('serialized_model.json', 'w') as fout:
#     fout.write(model_to_json(m))  # Save model
#
# with open('serialized_model.json', 'r') as fin:
#     m = model_from_json(fin.read())  # Load model


# Run to install Dask conda or pip
# conda install dask distributed
# python -m pip install "dask[distributed]" --upgrade
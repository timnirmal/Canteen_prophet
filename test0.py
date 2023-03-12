import pandas as pd
from matplotlib import pyplot
from prophet import Prophet

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

# add holidays
model = Prophet(holidays=holidays, changepoint_prior_scale=0.75, changepoint_range=0.8, yearly_seasonality=True,
                weekly_seasonality=True, daily_seasonality=True, seasonality_mode='multiplicative',
                seasonality_prior_scale=10.0, holidays_prior_scale=10.0, mcmc_samples=0, interval_width=0.80,
                uncertainty_samples=1000, stan_backend=None)

# fit the model
model.fit(df)

future = model.make_future_dataframe(periods=800, freq='D')

print("future")
print(future.head())

# use the model to make a forecast
forecast = model.predict(future)

# summarize the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

# plot forecast
model.plot(forecast)
pyplot.show()

# plot forecast components
model.plot_components(forecast)
pyplot.show()

from prophet.plot import add_changepoints_to_plot

fig = model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model, forecast)
pyplot.show()

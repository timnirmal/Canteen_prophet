import pandas as pd
from matplotlib import pyplot
from prophet import Prophet

from lib.datetimegen import generate_datetimes as dgt

# load the data/transposed first column
# df = pd.read_csv("data/transposed/canteenW_interval_counts_transposed.csv")
df = pd.read_csv("data/df1.csv")

# keep only the datetime and sales columns
df = df[["datetime", "sales"]]
# datetime column to datetime type
df["datetime"] = pd.to_datetime(df["datetime"])
# rename the columns
df.columns = ["ds", "y"]

print(df.dtypes)
print(df.head())

# define the model
model = Prophet()
# fit the model
model.fit(df)


future = dgt(30, "2022-12-23 14:15:00")

print(future.head())
# save to csv
future.to_csv("data/future_2.csv", index=False)


# future = model.make_future_dataframe(periods=12, freq='M')
# print
print(future.head())
# save to csv
future.to_csv("data/future.csv", index=False)


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

# save to csv
forecast.to_csv("data/forecast.csv", index=False)




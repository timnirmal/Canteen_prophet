# check prophet version
import pandas as pd
import prophet

# print version number
print('Prophet %s' % prophet.__version__)

# load the car sales dataset
from pandas import read_csv

# load data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv'
df = read_csv(path, header=0)
# summarize shape
print(df.shape)
# show first few rows
print(df.head())

# plot the dataset
from matplotlib import pyplot

df.plot()
pyplot.show()

# prepare expected column names
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

print(df.head())

# define the model
model = prophet.Prophet()
# fit the model
model.fit(df)

print(model)

# # define the period for which we want a prediction
# future = list()
# for i in range(1, 13):
#     date = '1968-%02d' % i
#     future.append([date])
# future = pd.DataFrame(future)
# future.columns = ['ds']
# future['ds'] = pd.to_datetime(future['ds'])

future = model.make_future_dataframe(periods=5000, freq='D')

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



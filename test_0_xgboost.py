import xgboost
import pandas as pd
print(xgboost.__version__)

from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from matplotlib import pyplot

# load every column in pd head
pd.set_option('display.max_columns', None)



# transform time series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]

# fit an xgboost model and make a one step prediction
def xgboost_forecast(train, test_X):
    # transform list into array
    train = array(train)
    # split into input and output columns
    train_X, train_y = train[:, :-1], train[:, -1]
    # fit model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
    model.fit(train_X, train_y)
    # make a one-step prediction
    yhat = model.predict([test_X])
    return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    # seed history with training dataset
    history = [x for x in train]
    # step over each time-step in the test set
    for i in range(len(test)):
        # split test row into input and output columns
        test_X, test_y = test[i, :-1], test[i, -1]
        # fit model on history and make a prediction
        yhat = xgboost_forecast(history, test_X)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.1f, predicted=%.1f' % (test_y, yhat))
    # estimate prediction error
    error = sqrt(mean_squared_error(test[:, -1], predictions))
    print('RMSE: %.3f' % error)
    # plot expected vs predicted
    pyplot.plot(test[:, -1])
    pyplot.plot(predictions)
    pyplot.show()



# load the data/transposed first column
df = pd.read_csv("data/df3.csv", header=0, index_col=0)

values = df.values
# plot each column
pyplot.figure()
for i in range(len(df.columns)):
    pyplot.subplot(len(df.columns), 1, i+1)
    pyplot.plot(values[:, i])
    pyplot.title(df.columns[i], y=0.5, loc='right')
pyplot.show()


# transform time series to supervised learning
reframed = series_to_supervised(df, n_in=6)

print(reframed.head())

# save to csv
reframed.to_csv("data/reframed-2.csv", index=False)

# evaluate
mae, y, yhat = walk_forward_validation(reframed, 12)
print('MAE: %.3f' % mae)

# plot expected vs predicted
pyplot.plot(y, label="Expected")
pyplot.plot(yhat , label="Predicted")
pyplot.legend()
pyplot.show()


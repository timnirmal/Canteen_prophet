import pandas as pd
from matplotlib import pyplot
from prophet import Prophet

from lib.datetimegen import generate_datetimes as dgt

# load the data/transposed first column
df = pd.read_csv("data/df1.csv")

# keep only the datetimes after 2020-01-01 00:00:00
df = df[df["ds"] >= "2020-01-01 00:00:00"]

# save to csv
df.to_csv("data/df3.csv", index=False)

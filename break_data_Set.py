import pandas as pd
from matplotlib import pyplot
from prophet import Prophet

from lib.datetimegen import generate_datetimes as dgt

# load the data/transposed first column
df = pd.read_csv("data/transposed/canteenW_interval_counts_transposed.csv")

# keep only the datetime and sales columns
df = df[["datetime", "sales"]]
# datetime column to datetime type
df["datetime"] = pd.to_datetime(df["datetime"])
# rename the columns
df.columns = ["ds", "y"]

# break the data into 2 sets by 2022-12-31 23:45:00
df1 = df[df["ds"] < "2022-12-31 23:45:00"]
df2 = df[df["ds"] >= "2022-12-31 23:45:00"]

# save to csv
df1.to_csv("data/df1.csv", index=False)
df2.to_csv("data/df2.csv", index=False)


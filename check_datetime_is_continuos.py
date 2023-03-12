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

# Datetime goes as 15 minute intervals
# if datetime is not continuous, remove the rows after the first datetime that is not continuous
# code
df = df[df["ds"].diff().dt.seconds == 900]
# print
print(df.head())

# save to csv
df.to_csv("data/df.csv", index=False)





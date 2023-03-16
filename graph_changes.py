# yearly_seasonality: auto
# weekly_seasonality: auto
# daily_seasonality: False
# growth: linear
# n_changepoints: 500
#
# Metrics
# R2: 0.020994221779617406
# MSE: 3858.073531503645
# MAE: 21.481064011293853
#
#
#
# yearly_seasonality: auto
# weekly_seasonality: True
# daily_seasonality: auto
# growth: linear
# n_changepoints: 25
#
# Metrics
# R2: 0.263759336557678
# MSE: 2901.382892353156
# MAE: 23.187593725324977
#
#
# yearly_seasonality: auto
# weekly_seasonality: True
# daily_seasonality: auto
# growth: linear
# n_changepoints: 50
#
# Metrics
# R2: 0.26566881751094285
# MSE: 2893.8579950659378
# MAE: 23.114650768344013
#
#
# yearly_seasonality: auto
# weekly_seasonality: True
# daily_seasonality: auto
# growth: linear
# n_changepoints: 100
#
# Metrics
# R2: 0.2629970958370851
# MSE: 2904.3867364715225
# MAE: 23.281544832104917
#
#
#
# yearly_seasonality: auto
# weekly_seasonality: True
# daily_seasonality: auto
# growth: linear
# n_changepoints: 200
#
# Metrics
# R2: 0.2650509893655796
# MSE: 2896.2927369925656
# MAE: 23.108590489212215
#
#
#
# yearly_seasonality: auto
# weekly_seasonality: True
# daily_seasonality: auto
# growth: linear
# n_changepoints: 500
#
# Metrics
# R2: 0.2623414608623127
# MSE: 2906.9704678434605
# MAE: 23.245369851477708
#
#
#
# yearly_seasonality: auto
# weekly_seasonality: True
# daily_seasonality: True
# growth: linear
# n_changepoints: 25
#
# Metrics
# R2: 0.263759336557678
# MSE: 2901.382892353156
# MAE: 23.187593725324977
#
#
#
# yearly_seasonality: auto
# weekly_seasonality: True
# daily_seasonality: True
# growth: linear
# n_changepoints: 50
#
# Metrics
# R2: 0.26566881751094285
# MSE: 2893.8579950659378
# MAE: 23.114650768344013
#
#
#
# yearly_seasonality: auto
# weekly_seasonality: True
# daily_seasonality: True
# growth: linear
# n_changepoints: 100
#
# Metrics
# R2: 0.2629970958370851
# MSE: 2904.3867364715225
# MAE: 23.281544832104917
#
#
#
# yearly_seasonality: auto
# weekly_seasonality: True
# daily_seasonality: True
# growth: linear
# n_changepoints: 200
#
# Metrics
# R2: 0.2650509893655796
# MSE: 2896.2927369925656
# MAE: 23.108590489212215


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# yealy_seasonality, weekly_seasonality, daily_seasonality, growth, n_changepoints, R2, MSE, MAE to dataframe
df = pd.DataFrame({'yearly_seasonality': [
    'auto', 'a
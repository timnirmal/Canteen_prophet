params = {
    # 'growth': 'linear',
    # 'changepoints': None,
    'n_changepoints': 50,
    "changepoint_range": 0.8,
    "yearly_seasonality": True,
    "weekly_seasonality": True,
    "daily_seasonality": True,
    'holidays': holidays,
    "seasonality_mode": "multiplicative",
    'seasonality_prior_scale': 10.0,
    'holidays_prior_scale': 10.0,
    "changepoint_prior_scale": 0.05,
    'mcmc_samples': 0,
    'interval_width': 0.8,
    'uncertainty_samples': 1000,
    'stan_backend': None,
}

Metrics
R2:  0.3817349882113993
MSE:  2436.463533210562
MAE:  18.771499024460585
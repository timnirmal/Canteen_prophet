import pandas as pd
from matplotlib import pyplot
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from prophet.serialize import model_to_json, model_from_json
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import numpy as np
import os
from prophet.plot import add_changepoints_to_plot

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

# param_grid = {
#     'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
#     # 'changepoint_prior_scale': [1, 0.5, 0.1, 0.05, 0.01, 0.001],
#     'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
#     # 'seasonality_prior_scale': [100, 10, 1, 0.1, 0.01],
#     'holidays': [holidays, None],
#     # 'holidays': [None, holidays],
#     'seasonality_mode': ['multiplicative'],
#     'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
#     # 'holidays_prior_scale': [100, 10, 1, 0.1, 0.01, 0],
#     'mcmc_samples': [0, 10, 100, 1000],
#     # 'mcmc_samples': [1000, 100, 10, 0],
#     'interval_width': [0.5, 0.8, 1],
#     # 'interval_width': [100, 10, 1, 0.5, 0],
#     'uncertainty_samples': [1000, 2000, 5000],
#     # 'uncertainty_samples': [5000, 2000, 1000],
#     'stan_backend': [None],
#     'changepoint_range': [0.8, 0.9, 1],
#     # 'changepoint_range': [1, 0.9, 0.8],
#     'yearly_seasonality': [True, False],
#     'weekly_seasonality': [True, False],
#     'daily_seasonality': [True, False],
#     # 'growth': ['linear', 'logistic'],
#     'growth': ['linear'],
#     'n_changepoints': [25, 50, 100, 200, 500],
#     # 'n_changepoints': [500, 200, 100, 50, 25],
#
# }


param_grid = {
    'n_changepoints': [50, 200],
    "changepoint_range": [0.8],
    "yearly_seasonality": [True],
    "weekly_seasonality": [True],
    "daily_seasonality": [True],
    'holidays': [holidays, None],
    "seasonality_mode": ["multiplicative"],
    'seasonality_prior_scale': [0.01, 10.0],
    'holidays_prior_scale': [0.01, 10.0],
    "changepoint_prior_scale": [0.001, 0.01, 0.05, 0.1, 0.5, 1, 10],
    'mcmc_samples': [0],
    'interval_width': [0.8],
    'uncertainty_samples': [1000],
    'stan_backend': [None],
}

# create results folder

if not os.path.exists("results"):
    os.makedirs("results")

folder_name = "prophet-results-" + str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S"))

# inside results folder create folder named  now date(YYYYMMDD)-time(HHMMSS)
if not os.path.exists("results/" + folder_name):
    os.makedirs("results/" + folder_name)

# create a file named results.txt
results_file = open("results/" + folder_name + "/results.txt", "w+")

# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters
for params in all_params:
    """LOG"""
    # variable for iteration number
    iteration = str(all_params.index(params))
    # iteration folder path
    iteration_path = "results/" + folder_name + "/" + iteration
    # inside results/now date(YYYYMMDD)-time(HHMMSS)-iteration folder create folder named iteration number
    if os.path != iteration_path:
        os.makedirs(iteration_path)

    # create results + iteration folder path + .txt file
    results_file_itr = open(iteration_path + "/results.txt", "w+")

    # write iteration number, datetime of run to folder_name/results.txt
    with open("results/" + folder_name + "/results.txt", "a") as myfile:
        myfile.write("\n\n============================================================\n")
        myfile.write("Iteration : " + iteration + "\n")
        myfile.write("Datetime : " + str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S")) + "\n\n")
    # write params to iteration_path/results.txt
    with open(iteration_path + "/results.txt", "a") as myfile:
        myfile.write("\n\n============================================================\n")
        myfile.write("Iteration : " + iteration + "\n")
        myfile.write("Datetime : " + str(pd.to_datetime('today').strftime("%Y%m%d-%H%M%S")) + "\n\n")

    # print params
    print("Params: ", params)
    # for each item in params json, write it to the file
    with open("results/" + folder_name + "/results.txt", "a") as results_file:
        results_file.write("Params: \n")
        for key, value in params.items():
            results_file.write("%s: %s\n" % (key, value))
        results_file.write("\n\n")
    with open(iteration_path + "/results.txt", "a") as results_file:
        results_file.write("Params: \n")
        for key, value in params.items():
            results_file.write("%s: %s\n" % (key, value))
        results_file.write("\n\n")

    print("\nIterration : ", iteration)
    print("Model Training...\n")
    m = Prophet(**params).fit(df)  # Fit model with given params
    # df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='365 days')

    print("\nCross Validation...\n")
    df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days')
    df_p = performance_metrics(df_cv)
    rmses.append(df_p['rmse'].values[0])

    """LOG"""
    # save df_cv to results folder with iteration number
    df_cv.to_csv(iteration_path + "/df_cv-" + iteration + ".csv", index=False)
    # save df_p to results folder
    df_p.to_csv(iteration_path + "/df_p-" + iteration + ".csv", index=False)

    future = m.make_future_dataframe(periods=200, freq='D')
    future.to_csv(iteration_path + "/future-" + iteration + ".csv", index=False)

    # use the model to make a forecast
    forecast = m.predict(future)
    forecast.to_csv(iteration_path + "/forecast-" + iteration + ".csv", index=False)

    r2_score(df["y"], forecast["yhat"][:len(df["y"])])
    mean_squared_error(df["y"], forecast["yhat"][:len(df["y"])])
    mean_absolute_error(df["y"], forecast["yhat"][:len(df["y"])])

    print("Metrics")
    print("R2: ", r2_score(df["y"], forecast["yhat"][:len(df["y"])]))
    print("MSE: ", mean_squared_error(df["y"], forecast["yhat"][:len(df["y"])]))
    print("MAE: ", mean_absolute_error(df["y"], forecast["yhat"][:len(df["y"])]))

    # Metrics to log
    with open("results/" + folder_name + "/results.txt", "a") as myfile:
        myfile.write("\n\nMetrics\n")
        myfile.write("R2: ")
        myfile.write(str(r2_score(df["y"], forecast["yhat"][:len(df["y"])])))
        myfile.write("\nMSE: ")
        myfile.write(str(mean_squared_error(df["y"], forecast["yhat"][:len(df["y"])])))
        myfile.write("\nMAE: ")
        myfile.write(str(mean_absolute_error(df["y"], forecast["yhat"][:len(df["y"])])))
        myfile.write("\n\n")
    with open(iteration_path + "/results.txt", "a") as myfile:
        myfile.write("\n\nMetrics\n")
        myfile.write("R2: ")
        myfile.write(str(r2_score(df["y"], forecast["yhat"][:len(df["y"])])))
        myfile.write("\nMSE: ")
        myfile.write(str(mean_squared_error(df["y"], forecast["yhat"][:len(df["y"])])))
        myfile.write("\nMAE: ")
        myfile.write(str(mean_absolute_error(df["y"], forecast["yhat"][:len(df["y"])])))
        myfile.write("\n\n")

    # save model to json file in results folder/iteration folder
    with open(iteration_path + '/serialized_model-' + iteration + '.json', 'w') as fout:
        fout.write(model_to_json(m))  # Save model

    # plot forecast
    m.plot(forecast)
    # save plot to image
    pyplot.savefig(iteration_path + "/forecast-" + iteration + ".png")

    # plot forecast components
    m.plot_components(forecast)
    # save plot to image
    pyplot.savefig(iteration_path + "/forecast_components-" + iteration + ".png")

    # plot changepoints
    fig = m.plot(forecast)
    a = add_changepoints_to_plot(fig.gca(), m, forecast)
    # save plot to image
    pyplot.savefig(iteration_path + "/changepoints-" + iteration + ".png")

# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)

best_params = all_params[np.argmin(rmses)]
print('Best params: ', best_params)

with open("results/" + folder_name + "/results.txt", "a") as myfile:
    myfile.write("\n\nFinal Result\n")
    myfile.write(str(tuning_results))
    myfile.write('Best params: ')
    myfile.write(str(best_params))

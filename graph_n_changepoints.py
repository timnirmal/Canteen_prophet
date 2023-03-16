# n_changepoints: 25
# R2: 0.263759336557678
# MSE: 2901.382892353156
# MAE: 23.187593725324977
#
# n_changepoints: 50
# R2: 0.26566881751094285
# MSE: 2893.8579950659378
# MAE: 23.114650768344013
#
# n_changepoints: 100
# R2: 0.2629970958370851
# MSE: 2904.3867364715225
# MAE: 23.281544832104917
#
# n_changepoints: 200
# R2: 0.2650509893655796
# MSE: 2896.2927369925656
# MAE: 23.108590489212215
#
# n_changepoints: 500
# R2: 0.2623414608623127
# MSE: 2906.9704678434605
# MAE: 23.245369851477708

# graph R2, MSE, MAE for each n_changepoints

import matplotlib.pyplot as plt
import numpy as np

n_changepoints = [25, 50, 100, 200, 500]
R2 = [0.263759336557678, 0.26566881751094285, 0.2629970958370851, 0.2650509893655796, 0.2623414608623127]
MSE = [2901.382892353156, 2893.8579950659378, 2904.3867364715225, 2896.2927369925656, 2906.9704678434605]
MAE = [23.187593725324977, 23.114650768344013, 23.281544832104917, 23.108590489212215, 23.245369851477708]


# plot R2
plt.plot(n_changepoints, R2, 'o-')
plt.title('R2')
plt.xlabel('n_changepoints')
plt.ylabel('R2')
plt.show()

# plot MSE
plt.plot(n_changepoints, MSE, 'o-')
plt.title('MSE')
plt.xlabel('n_changepoints')
plt.ylabel('MSE')
plt.show()

# plot MAE
plt.plot(n_changepoints, MAE, 'o-')
plt.title('MAE')
plt.xlabel('n_changepoints')
plt.ylabel('MAE')
plt.show()



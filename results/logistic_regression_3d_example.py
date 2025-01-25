import plotly
import plotly.graph_objs as go
import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from mpl_toolkits.mplot3d import Axes3D

log_file = "logs_3_dec"
log_directory = os.path.join(os.path.expanduser('~'), f"sim_loop/results/logs/{log_file}")
output_file = f'{log_directory}/compiled_log_data.csv'
df_3 = pd.read_csv(output_file)

log_file = "logs_5_dec"
log_directory = os.path.join(os.path.expanduser('~'), f"sim_loop/results/logs/{log_file}")
output_file = f'{log_directory}/compiled_log_data.csv'
df_5 = pd.read_csv(output_file)

log_file = "logs_7_dec"
log_directory = os.path.join(os.path.expanduser('~'), f"sim_loop/results/logs/{log_file}")
output_file = f'{log_directory}/compiled_log_data.csv'
df_7 = pd.read_csv(output_file)

log_directory = os.path.join(os.path.expanduser('~'), "sim_loop/results/CC_human_driver_cut_in")
output_file = f'{log_directory}/CC_human_driver_cut_in.csv'
df_cc_human_driver = pd.read_csv(output_file)

feature_cols = ['s_delta', 'v_delta', 'ego_max_dec']
X = df_cc_human_driver[feature_cols] # Features
Y = df_cc_human_driver.collision # Target variable

# Fit the data to a logistic regression model.
clf = sklearn.linear_model.LogisticRegression()
clf.fit(X, Y)

# The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
# Solve for w3 (z)
z = lambda x,y: (-clf.intercept_[0]-clf.coef_[0][0]*x -clf.coef_[0][1]*y) / clf.coef_[0][2]

tmp_x = np.linspace(10,60,60)
tmp_y = np.linspace(10,40,60)
x,y = np.meshgrid(tmp_x,tmp_y)
z_values = z(x,y)

# Define thresholds & mask
a = 7  # Example threshold for upper limit
b = 3   # Example threshold for lower limit
mask = (z_values > a) | (z_values < b)

# Set excluded values to np.nan (or any other value)
z_values[mask] = np.nan

df_coll = df_cc_human_driver[~df_cc_human_driver['collision']]
df_no_coll = df_cc_human_driver[df_cc_human_driver['collision']]


# Assuming clf is already defined and fitted
z_target = 5
x_values = np.linspace(10, 60, 30)
y_values = (-clf.intercept_[0] - clf.coef_[0][0] * x_values - z_target * clf.coef_[0][2]) / clf.coef_[0][1]
points = list(zip(x_values, y_values))


#from scipy.stats import skewnorm
#a=10
#data= skewnorm.rvs(a, size=1000)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot3D(df_coll['s_delta'], df_coll['v_delta'], df_coll['ego_max_dec'],'ob')
ax.plot3D(df_no_coll['s_delta'], df_no_coll['v_delta'], df_no_coll['ego_max_dec'],'sr')
ax.plot_surface(x, y, z_values)
ax.scatter3D(x_values, y_values, z_target)
ax.view_init(30, 60)
plt.show()
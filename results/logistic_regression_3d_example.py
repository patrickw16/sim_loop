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
from scipy.stats import skewnorm

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

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')

def filter_points(points, x_thresholds, y_thresholds, z_thresholds):

    # Unpack thresholds
    x_low, x_high = x_thresholds
    y_low, y_high = y_thresholds
    z_low, z_high = z_thresholds

    # Create boolean masks for each dimension
    mask_x = (points[:, 0] >= x_low) & (points[:, 0] <= x_high)
    mask_y = (points[:, 1] >= y_low) & (points[:, 1] <= y_high)
    mask_z = (points[:, 2] >= z_low) & (points[:, 2] <= z_high)

    # Combine masks to filter points
    valid_mask = mask_x & mask_y & mask_z
    filtered_points = points[valid_mask]

    return filtered_points

def plot_skew_normal_with_noise(clf, z_target):

    # Generate x values
    x_values = np.linspace(10, 60, 2000)

    y_values = (-clf.intercept_[0] - clf.coef_[0][0] * x_values - z_target * clf.coef_[0][2]) / clf.coef_[0][1]

    m, c = np.polyfit(x_values, y_values, 1)

    a = -1  # skewness parameter
    scale = 10  # scale parameter to adjust variance
    loc = 5  # location parameter

    y_noise = skewnorm.rvs(a=a, loc=loc, scale=scale, size=len(x_values))
    y_with_noise = (m * x_values + c) + y_noise
    z_with_noise = z_target + np.random.normal(0, np.sqrt(0.05), len(x_values))

    ax.scatter3D(x_values, y_values, z_target, label='Original Points', color='black')

    noisy_points = np.column_stack((x_values, y_with_noise, z_with_noise))

    return noisy_points, a, scale, loc

all_noisy_points = []

for z_target in np.arange(3,8,1):
    noisy_points, a, scale, loc = plot_skew_normal_with_noise(clf, z_target=z_target)
    all_noisy_points.append(noisy_points)

combined_noisy_points = np.vstack(all_noisy_points)
print(combined_noisy_points.shape)

# Define thresholds for filtering
x_thresholds = (10, 60)  # Example thresholds for x (s_delta)
y_thresholds = (10, 40)  # Example thresholds for y (v_delta)
z_thresholds = (3, 7)     # Example thresholds for z (ego_max_dec)

# Filter the combined noisy points
filtered_noisy_points = filter_points(combined_noisy_points, x_thresholds, y_thresholds, z_thresholds)
print(filtered_noisy_points.shape)

np.save(f'prior_points_s_delta_v_delta_ego_max_dec_{a}_{scale}_{loc}.npy', filtered_noisy_points)

ax.plot3D(df_coll['s_delta'], df_coll['v_delta'], df_coll['ego_max_dec'],'ob')
ax.plot3D(df_no_coll['s_delta'], df_no_coll['v_delta'], df_no_coll['ego_max_dec'],'sr')
ax.scatter3D(filtered_noisy_points[:,0], filtered_noisy_points[:,1], filtered_noisy_points[:,2], color='yellow')
ax.plot_surface(x, y, z_values)

ax.view_init(30, 60)
plt.show()
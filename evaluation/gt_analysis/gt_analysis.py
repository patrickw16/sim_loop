import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

log_file = "logs_3_dec"
log_directory = os.path.join(os.path.expanduser('~'), f"sim_loop/results/logs/{log_file}")
output_file = f'{log_directory}/compiled_log_data.csv'
df_3 = pd.read_csv(output_file)
df_3['ego_max_dec'] = 3

log_file = "logs_4_dec"
log_directory = os.path.join(os.path.expanduser('~'), f"sim_loop/results/logs/{log_file}")
output_file = f'{log_directory}/compiled_log_data.csv'
df_4 = pd.read_csv(output_file)
df_4['ego_max_dec'] = 4

log_file = "logs_5_dec"
log_directory = os.path.join(os.path.expanduser('~'), f"sim_loop/results/logs/{log_file}")
output_file = f'{log_directory}/compiled_log_data.csv'
df_5 = pd.read_csv(output_file)
df_5['ego_max_dec'] = 5

log_file = "logs_6_dec"
log_directory = os.path.join(os.path.expanduser('~'), f"sim_loop/results/logs/{log_file}")
output_file = f'{log_directory}/compiled_log_data.csv'
df_6 = pd.read_csv(output_file)
df_6['ego_max_dec'] = 6

log_file = "logs_7_dec"
log_directory = os.path.join(os.path.expanduser('~'), f"sim_loop/results/logs/{log_file}")
output_file = f'{log_directory}/compiled_log_data.csv'
df_7 = pd.read_csv(output_file)
df_7['ego_max_dec'] = 7

dataframes =  [df_3, df_4, df_5, df_6, df_7]
# Combine all DataFrames into a single DataFrame
df = pd.concat(dataframes, ignore_index=True)


##### Generate data for evaluation purposes #####

#[s_delta, v_delta, ego_max_dec]

#cube_sizes
s_delta_range = np.arange(10,62,2)
v_delta_range = np.arange(10,42,2)
ego_max_dec_range = np.arange(3,8,1)

if len(ego_max_dec_range)*len(s_delta_range)*len(v_delta_range) != len(df):
    print("Warning: Sizes do not match")

gt_cube_sizes = np.array([np.ones(2080)*1, np.ones(2080)*1, np.ones(2080)*0.5]).T
np.save('evaluation/gt_analysis/data/gt_cube_sizes', gt_cube_sizes)

#sub_cube_centers
gt_sub_cube_centers = np.array([df['s_delta'], df['v_delta'], df['ego_max_dec']]).T
np.save('evaluation/gt_analysis/data/gt_sub_cube_centers', gt_sub_cube_centers)

#gt_pass_fail
gt_pass_fail = df['CollisionDetected'].to_list()
np.save('evaluation/gt_analysis/data/gt_pass_fail', gt_pass_fail)

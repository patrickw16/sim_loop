import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Generate a DataFrame with evenly spaced points
x = np.arange(0, 5, 1)  # Example: x values from 0 to 4 with step 1
y = np.arange(0, 5, 1)  # Example: y values from 0 to 4 with step 1
z = np.arange(0, 5, 1)  # Example: z values from 0 to 4 with step 1

# Create a grid of points
xx, yy, zz = np.meshgrid(x, y, z)
df = pd.DataFrame({'x': xx.flatten(), 'y': yy.flatten(), 'z': zz.flatten()})


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

# Size of each small cube (half-length of the cube's side)
cube_size = 0.5

# Function to define the vertices of a cube centered at a given point
def create_cube(center, size):
    x, y, z = center
    vertices = np.array([
        [x - size, y - size, z - size],  # Vertex 1
        [x + size, y - size, z - size],  # Vertex 2
        [x + size, y + size, z - size],  # Vertex 3
        [x - size, y + size, z - size],  # Vertex 4
        [x - size, y - size, z + size],  # Vertex 5
        [x + size, y - size, z + size],  # Vertex 6
        [x + size, y + size, z + size],  # Vertex 7
        [x - size, y + size, z + size]   # Vertex 8
    ])
    return vertices

# Function to define the edges of a cube
def create_edges():
    return np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
    ])

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))  # Increase figure size for better visualization
ax = fig.add_subplot(111, projection='3d')

# Plot each small cube
for _, row in df.iterrows():
    center = (row['v_delta'], row['s_delta'], row['ego_max_dec'])
    vertices = create_cube(center, cube_size)
    edges = create_edges()
    
    if row['v_delta'] == 28 and row['s_delta'] == 52 and row['ego_max_dec'] == 3:
        color = 'black'
        alpha = 1.0
    else:
        color = 'b'
        alpha = 0.1

    # Plot the edges of the cube
    for edge in edges:
        ax.plot(vertices[edge, 0], vertices[edge, 1], vertices[edge, 2], c=color, alpha=alpha)  # Use transparency

# Plot the original points from the DataFrame
ax.scatter(df['v_delta'], df['s_delta'], df['ego_max_dec'], c='r', marker='o', s=20, label='Centers', alpha=0.8)

# Set the limits of the axes
ax.set_xlim([df['v_delta'].min() - 1, df['v_delta'].max() + 1])
ax.set_ylim([df['s_delta'].min() - 1, df['s_delta'].max() + 1])
ax.set_zlim([df['ego_max_dec'].min() - 1, df['ego_max_dec'].max() + 1])

# Set the labels of the axes
ax.set_xlabel('v_delta (km/h)')
ax.set_ylabel('s_delta (m)')
ax.set_zlabel('ego_max_dec (m/s²)')

# Add a legend
ax.legend()

# Show the plot
plt.show()
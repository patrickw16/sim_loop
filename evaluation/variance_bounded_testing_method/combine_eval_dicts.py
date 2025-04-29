import pandas as pd
import glob

# Specify the path to your CSV files
path = 'evaluation/variance_bounded_testing_method/data/eval_dicts/'  # Update this to your directory
all_files = glob.glob(path + "*.csv")  # Get all CSV files in the directory

# Create an empty list to hold the DataFrames
dataframes = []

# Loop through the list of files and read each one into a DataFrame
for filename in all_files:
    df = pd.read_csv(filename)  # Read the CSV file
    dataframes.append(df)  # Append the DataFrame to the list

# Concatenate all DataFrames in the list into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Optionally, you can save the combined DataFrame to a new CSV file
combined_df.to_csv('variance_bounded_testing_method_evaluation_dict.csv', index=False)

# Display the combined DataFrame
print(combined_df)
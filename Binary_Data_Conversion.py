import os
import pandas as pd
import h5py

# Define the directory containing the data files
data_directory = r'C:\Users\lolaf\Desktop\Thesis_code\Data'

# Load the training and testing data directly using the file paths
train_file_path = os.path.join(data_directory, 'Y_train.csv')
test_file_path = os.path.join(data_directory, 'Y_test.csv')

# Load the datasets without headers
train_df = pd.read_csv(train_file_path, header=None)
test_df = pd.read_csv(test_file_path, header=None)

# Create a new 'drone' column where it is 1 if any of the drone signal columns (0-7, 9) are 1, else 0
drone_columns = [0, 1, 2, 3, 4, 5, 6, 7, 9]
train_df['drone'] = train_df[drone_columns].max(axis=1)
test_df['drone'] = test_df[drone_columns].max(axis=1)

# Keep the Wi-Fi column as is
train_df['wifi'] = train_df.iloc[:, 8]
test_df['wifi'] = test_df.iloc[:, 8]

# Select only the new 'drone' and 'wifi' columns
train_df = train_df[['drone', 'wifi']]
test_df = test_df[['drone', 'wifi']]

# Save the modified data to a new HDF5 file
h5_file_path = os.path.join(data_directory, 'binary_test_train_values.h5')
with h5py.File(h5_file_path, 'w') as h5f:
    h5f.create_dataset('Y_train', data=train_df.to_numpy())
    h5f.create_dataset('Y_test', data=test_df.to_numpy())

print(f"Conversion complete. The binary classified data is saved to 'binary_test_train_values.h5'.")

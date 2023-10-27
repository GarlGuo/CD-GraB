import os
import subprocess
import numpy as np
import pandas as pd

# Download M4 data from https://github.com/Mcompetitions/M4-methods/tree/master/Dataset
for freq in ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]:
    subprocess.call(
        ['wget', 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/' + freq + '-train.csv'])
    subprocess.call(
        ['wget', 'https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Test/' + freq + '-test.csv'])


# Load M4 csv data
train_data, test_data = {}, {}
freq_lst = ["Hourly", "Daily", "Weekly", "Monthly", "Quarterly", "Yearly"]
for freq in freq_lst:
    train_data[freq] = pd.read_csv(freq + '-train.csv')
    test_data[freq] = pd.read_csv(freq + '-test.csv')

# Extract time series from pandaframe and remove NaN
train_np, test_np = {}, {}
for freq in freq_lst:
    temp_lst = []
    for i in range(train_data[freq].values.shape[0]):
        temp_lst.append(
            np.array(train_data[freq].iloc[i, 1:].dropna().values, dtype=float))
    train_np[freq] = temp_lst

    temp_lst = []
    for i in range(test_data[freq].values.shape[0]):
        temp_lst.append(
            np.array(test_data[freq].iloc[i, 1:].dropna().values, dtype=float))
    test_np[freq] = temp_lst

if not os.path.exists(f'data{os.sep}M4'):
    os.makedirs(f'data{os.sep}M4')


# Save data dictionaries into numpy files
np.save(f"data{os.sep}M4{os.sep}train.npy", train_np)
np.save(f"data{os.sep}M4{os.sep}test.npy", test_np)

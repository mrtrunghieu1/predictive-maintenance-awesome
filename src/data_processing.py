import os
import pandas as pd
import numpy as np
from configs.configs import *
from sklearn.preprocessing import MinMaxScaler


# Extract time window sequences
def generate_sequences(group_train, i, window_length):
    # Extract all columns of each group without RUL column.
    data_each_group = group_train.get_group(i).iloc[:, :-1].values
    num_samples_each_group = data_each_group.shape[0]
    subtraction_samples = num_samples_each_group - window_length

    for start, stop in zip(range(0, subtraction_samples), range(window_length, num_samples_each_group)):
        yield data_each_group[start:stop, :]


# Extract label RUL for each time window
def generate_labels(group_train, i, window_length):
    # Get the RUL column
    labels_each_group = group_train.get_group(i).iloc[:, -1].values
    num_labels_each_group = labels_each_group.shape[0]

    return labels_each_group[window_length: num_labels_each_group]


def load_data(file_path):
    # define column names for easy indexing
    col_names = index_names + setting_names + sensor_names
    # read data
    train = pd.read_csv(os.path.join(dir_path, f"train_{file_path}.txt"), sep='\s+', header=None, names=col_names)
    test = pd.read_csv(os.path.join(dir_path, f"test_{file_path}.txt"), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv(os.path.join(dir_path, f"RUL_{file_path}.txt"), sep='\s+', header=None, names=['RUL'])

    # drop non-informative features in training set
    drop_sensors = ['sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    drop_labels = setting_names + drop_sensors
    train.drop(labels=drop_labels, axis=1, inplace=True)

    # separate title info and sensor data
    title = train.iloc[:, 0:2]
    data = train.iloc[:, 2:]

    # MinMax normalization
    min_max_scaler = MinMaxScaler()
    data_norm = pd.DataFrame(
        min_max_scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
    )
    train_norm = pd.concat([title, data_norm], axis=1)

    # Add remaining useful life (RUL)
    train_norm = add_RUL(train_norm)

    # Group the training set with unit
    group_train = train_norm.groupby(by="unit")

    # Drop non-informative features in the testing set
    test.drop(labels=drop_labels, axis=1, inplace=True)
    title = test.iloc[:, 0:2]
    data = test.iloc[:, 2:]
    data_norm = pd.DataFrame(
        min_max_scaler.fit_transform(data),
        columns=data.columns,
        index=data.index
    )
    test_norm = pd.concat([title, data_norm], axis=1)

    # Group the testing set with unit
    group_test = test_norm.groupby(by="unit")

    print("train_norm >> ", train_norm.shape)
    print("test_norm >> ", test_norm.shape)

    # Generate data
    sequences_extraction = (list(generate_sequences(group_train, i, window_length)) for i in
                            train_norm['unit'].unique())
    window_sequences = np.concatenate(list(sequences_extraction)).astype(np.float32)
    # Generate labels
    labels_extraction = list(generate_labels(group_train, i, window_length) for i in train_norm['unit'].unique())
    window_labels = np.concatenate(labels_extraction).reshape(-1, 1).astype(np.float32)

    return window_sequences, window_labels, group_train, group_test, y_test


def add_RUL(df):
    # Get the total number of cycles of each unit
    grouped_by_unit = df.groupby(by="unit")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name="max_cycle"), left_on="unit", right_index=True)

    # Calculate RUL for each row
    RUL = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = RUL

    # Drop max_cycle as it's not necessary
    result_frame.drop("max_cycle", axis=1, inplace=True)
    return result_frame


if __name__ == "__main__":
    file_path = 'FD001'
    load_data(file_path=file_path)

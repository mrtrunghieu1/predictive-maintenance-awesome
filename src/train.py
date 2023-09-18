import os
import torch
import argparse
from configs.configs import *
from data_processing import load_data


def training(group_train, group_test, y_test):
    for epoch in range(epochs):
        i = 1
        epoch_loss = 0

        # Iteration of each unit
        while i <= 100:
            # Fetch the data of unit i
            x = group_train.get_group(i).shape[1]
            print(x)
            total_loss = 0
            # optim.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FD001')
    args = parser.parse_args()

    # Loading training and testing sets
    window_sequences, window_labels, group_train, group_test, y_test = load_data(args.dataset)

    # Define and load model
    # model = LSTM()

    # Initialization

    # Initialize Adam optimizer
    # optimizer = torch.optim.Adam(lr=0.001)

    # Mean-squared error
    criterion = torch.nn.MSELoss()

    # Training with evaluation
    result, mse = training(group_train, group_test, y_test)

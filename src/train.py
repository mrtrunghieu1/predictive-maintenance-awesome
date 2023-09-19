import os
import torch
import argparse
from configs.configs import *
from data_processing import load_data
from models.LSTM import LSTM
from torch.autograd import Variable
import numpy as np

def testing_function(num_test, group_test):
    rmse_test, result_test = 0, list()
    for i in range(1, num_test + 1):
        X_test = group_test.get_group(i).iloc[:, 2:]
        X_test_tensors = Variable(torch.Tensor(X_test.to_numpy()))
        X_test_tensors = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

        test_predict = model.forward(X_test_tensors)
        data_predict = max(test_predict[-1].detach().numpy(), 0)
        result_test.append(data_predict)
        rmse_test = np.add(np.power((data_predict - y_test.to_numpy()[i - 1]), 2), rmse_test)

    rmse_test = (np.sqrt(rmse_test / num_test)).item()
    return result_test, rmse_test


def train(model, num_train, group_train):
    rmse_temp = 100
    for epoch in range(1, EPOCHS+1):

        model.train()
        epoch_loss = 0

        for i in range(1, num_train + 1):
            X, y = group_train.get_group(i).iloc[:, 2:-1], group_train.get_group(i).iloc[:, -1:]

            X_train_tensors = Variable(torch.Tensor(X.to_numpy()))
            y_train_tensors = Variable(torch.Tensor(y.to_numpy()))
            X_train_tensors = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))

            outputs = model(X_train_tensors)

            optimizer.zero_grad()  # calculate the gradient, manually setting to 0
            loss = criterion(outputs, y_train_tensors)  # obtain the loss function
            epoch_loss += loss.item()
            loss.backward()  # calculates the loss of the loss function
            optimizer.step()  # improve from loss, i.e back propagation

        if epoch % 1  == 0:
            # Evaluate model
            model.eval()
            result, rmse = testing_function(num_test, group_test)

            if rmse_temp < rmse and rmse_temp < 25:
                result, rmse = result_temp, rmse_temp
                break

            rmse_temp, result_temp = rmse, result  # store the last rmse
            print("Epoch: %d, loss: %1.5f, rmse: %1.5f" % (epoch, epoch_loss / num_train, rmse))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FD001')
    args = parser.parse_args()

    # Loading training and testing sets
    # window_sequcens = [num_windows, WINDOW_LENGTH, num_features]
    # window_labels = [num_windows, RUL_feature]
    window_sequences, window_labels, group_train, group_test, y_test = load_data(args.dataset)

    # Get number of training features without unit, time_in_cycles, RUL
    num_train, num_test = len(group_train.size()), len(group_test.size())
    num_features = group_train.get_group(1).shape[1] - 3

    # Define LSTM model
    model = LSTM(
        input_size=num_features,
        hidden_size=N_HIDDEN,
        num_layers=N_LAYER,
    )

    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training with evaluation
    train(model, num_train, group_train)

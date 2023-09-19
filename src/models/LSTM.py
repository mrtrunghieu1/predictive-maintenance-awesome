import torch
import torch.nn as nn
from torch.autograd import Variable

class LSTM(nn.Module):
    """
    LSTM architecture
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1, seq_length=30):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        # FC_1
        self.fc_1 = nn.Linear(hidden_size, 16)
        # FC_2
        self.fc_2 = nn.Linear(16, 8)
        # FC_3
        self.fc_3 = nn.Linear(8, 1)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        :param x: input_features
        :return: prediction features
        """

        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # (num_layers, batch_size, hidden_size)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # (num_layers, batch_size, hidden_size)

        # output will be (sequence_length, batch_size, hidden_size)
        output, (hn, cn) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        hn_o = torch.Tensor(hn.detach().numpy()[-1, :, :])
        hn_o = hn_o.view(-1, self.hidden_size)
        hn_1 = torch.Tensor(hn.detach().numpy()[1, :, :])
        hn_1 = hn_1.view(-1, self.hidden_size)

        out = self.relu(self.fc_1(self.relu(hn_o + hn_1)))
        out = self.relu(self.fc_2(out))
        out = self.dropout(out)
        out = self.fc_3(out)
        return out
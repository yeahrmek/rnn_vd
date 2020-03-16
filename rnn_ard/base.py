import math
import torch
from cplxmodule.relevance import LinearARD

import torch
from cplxmodule.relevance import LinearARD


class RNNBase(torch.nn.Module):
    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0., bidirectional=False):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.reset_parameters()

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, input, hidden):
        raise NotImplementedError

    def from_state_dict(self, state_dict):
        """
        Load weights from usual GRU/LSTM model

        Args:
            state_dict: dict
                State dictionary of the GRU/LSTM model

        Returns:

        """


class RNNCell(torch.nn.Module):
    def __init__(self, mode, input_size, hidden_size):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size

        if mode == 'LSTM':
            self.n_weights = 4
        elif mode == 'GRU':
            self.n_weights = 3


        self.weight_ih = torch.nn.Parameter(
                torch.randn(self.n_weights * hidden_size, input_size))
        self.weight_hh = torch.nn.Parameter(
                torch.randn(self.n_weights * hidden_size, hidden_size))
        self.bias_ih = torch.nn.Parameter(
                torch.randn(self.n_weights * hidden_size))
        self.bias_hh = torch.nn.Parameter(
                torch.randn(self.n_weights * hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if 'bias_hh' in name:
                if self.mode == 'GRU':
                    self.bias_hh.data[:self.hidden_size] = 1
                elif self.mode == 'LSTM':
                    self.bias_hh[self.hidden_size:self.hidden_size * 2] = 1
            else:
                torch.nn.init.uniform_(weight, -stdv, stdv)


    def forward(self, input, state):
        raise NotImplementedError

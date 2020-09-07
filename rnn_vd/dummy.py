import math

import torch
from cplxmodule.nn.masked import LinearMasked

from .rnn import RNNLayer, BidirRNNLayer
from .vardropout import LinearVD


class BaseDummyCell(torch.nn.Module):
    """
    Base class for dummy RNN cell, i.e. the hidden state is not used
    """
    def __init__(self, input_size, hidden_size, bias, linear,
                 activation, layernorm, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layernorm = layernorm

        self.linear_ih = linear(input_size, hidden_size, bias=bias)
        self.activation = activation(**kwargs)

        if layernorm:
            self.layernorm_ih = torch.nn.LayerNorm(hidden_size)

    def forward(self, inputs, hidden_state, **kwargs):
        nx = self.linear_ih(inputs)

        if self.layernorm:
            nx = self.layernorm_ih(nx)

        hidden_state = self.activation(nx)
        return hidden_state, hidden_state


class DummyCell(BaseDummyCell):
    """
    Dummy RNN cell, i.e. the hidden state is not used
    """
    def __init__(self, input_size, hidden_size, bias,
                 activation, layernorm=False, **kwargs):
        super().__init__(input_size, hidden_size, bias, torch.nn.Linear,
                         activation, layernorm, **kwargs)


class DummyVDCell(BaseDummyCell):
    """
    Dummy RNN cell with Variational Dropout, i.e. the hidden state is not
    used and `LinearVD` is used for fully connected layer
    """
    def __init__(self, input_size, hidden_size, bias,
                 activation, layernorm=False, **kwargs):
        super().__init__(input_size, hidden_size, bias, LinearVD,
                         activation, layernorm, **kwargs)

    def forward(self, inputs, hidden_state, **kwargs):
        # update dropout noise
        if self.training:
            self.linear_ih.update_weight_noise()

        return super().forward(inputs, hidden_state, **kwargs)


class DummyMaskedCell(BaseDummyCell):
    """
    Dummy RNN cell with Masked Linear layer, i.e. the hidden state is
    not  used and `LinearMasked` is used for fully connected layer
    """
    def __init__(self, input_size, hidden_size, bias,
                 activation, layernorm=False, **kwargs):
        super().__init__(input_size, hidden_size, bias, LinearMasked,
                         activation, layernorm, **kwargs)


class BaseDummyRNN(torch.nn.Module):
    """
    Dummy RNN, i.e. without hidden state

    Parameters:
    -----------
    input_size : int
        Dimensionality of input

    hidden_size : list
        List of hidden_sizes

    bias : bool
        Use bias flag

    activation : torch.nn.Module
        Activation class

    layernorm : bool, default=True
        Use layernorm flag

    **kwargs : dict
        Other arguments
    """
    def __init__(self, cell, input_size, hidden_size, num_layers=1,
                 bias=True, batch_first=False, dropout=0.,
                 bidirectional=False, activation=torch.nn.Tanh, **kwargs):
        super().__init__()

        assert batch_first == True

        self.cell = cell
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.activation = activation

        self.n_dirs = bidirectional + 1

        first_cell_args = (input_size, hidden_size[0], bias, activation)
        other_cell_args = [
            (in_size * self.n_dirs, h_size, bias, activation)
            for in_size, h_size in zip(hidden_size[:-1], hidden_size[1:])
        ]

        layer = BidirRNNLayer if bidirectional else RNNLayer
        self.layers = torch.nn.ModuleList(
                [layer(cell, *first_cell_args, **kwargs)] +
                [layer(cell, *other_cell_args[i], **kwargs)
             for i in range(num_layers - 1)]
        )
        if dropout > 0:
            self.dropout_layer = torch.nn.Dropout(dropout)

    def forward(self, inputs, hidden_state=None, **kwargs):
        output = inputs

        for i, rnn_layer in enumerate(self.layers):
            h = torch.zeros(inputs.shape[0], rnn_layer.cell.hidden_size)
            output, _ = rnn_layer(output, h, **kwargs)
            if i < self.num_layers - 1 and self.dropout > 0:
                output = self.dropout_layer(output)

        return output, None


class DummyRNN(BaseDummyRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(DummyCell, *args, **kwargs)


class DummyRNNVD(BaseDummyRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(DummyVDCell, *args, **kwargs)


class DummyRNNMasked(BaseDummyRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(DummyMaskedCell, *args, **kwargs)

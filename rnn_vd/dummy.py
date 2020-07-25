import math
import torch

from cplxmodule.nn.masked import LinearMasked

from .vardropout import LinearVD
from .rnn import BaseRNN


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


class DummyRNN(BaseRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(DummyCell, *args, **kwargs)


class DummyRNNVD(BaseRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(DummyVDCell, *args, **kwargs)


class DummyRNNMasked(BaseRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(DummyMaskedCell, *args, **kwargs)

from functools import partial

import torch
from t3nsor import TRLinear

from .rnn import BaseCell, GRUCellMixin, RNNMixin, BaseRNN


class GRUTRCell(GRUCellMixin, BaseCell):
    def __init__(self, input_size, hidden_size, bias,
                 activation=torch.nn.Tanh, **kwargs):
        linear = partial(TRLinear, naive=True)
        super().__init__('GRU', input_size, hidden_size, bias, linear,
                         activation, **kwargs)


class GRUTR(RNNMixin, BaseRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(GRUTRCell, *args, **kwargs)

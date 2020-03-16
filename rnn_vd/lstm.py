import torch
from .rnn import RNNBase
from cplxmodule.relevance import LinearARD


class LSTM(RNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__('LSTM', *args, **kwargs)

    def forward(self, input, hidden_state):
        raise NotImplementedError

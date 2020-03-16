import torch
from .base import RNNBase
from cplxmodule.relevance import LinearARD


class LSTM(RNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__('LSTM', *args, **kwargs)

    def forward(self, input, hidden):
        raise NotImplementedError

import torch
import torch.nn.functional as F

from .base import RNNBase, RNNCell
from cplxmodule.relevance import LinearARD


class GRU(RNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__('GRU', *args, **kwargs)

    def forward(self, input, hidden):
        raise NotImplementedError


class GRUCell(RNNCell):
    def __init__(self, input_size, hidden_size):
        super().__init__('GRU', input_size, hidden_size)

    def forward(self, input, hidden_state):
        gates_input = F.linear(input, self.weight_ih, bias=self.bias_ih)
        gates_hidden = F.linear(hidden_state, self.weight_hh, bias=self.bias_hh)

        rx, zx, nx = gates_input.chunk(self.n_weights, 1)
        rh, zh, nh = gates_hidden.chunk(self.n_weights, 1)

        r = F.sigmoid(rx + rh)
        z = F.sigmoid(zx + zh)
        n = F.tanh(nx + r * nh)

        hidden_state = (1 - z) * n + z * hidden_state
        return hidden_state


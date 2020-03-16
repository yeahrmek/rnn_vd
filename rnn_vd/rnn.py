import math
import torch
import torch.nn.functional as F


from .varropout import LinearVD


class RNNBase(torch.nn.Module):
    def __init__(self, mode, input_size, hidden_size,
                 num_layers=1, bias=True, batch_first=False,
                 dropout=0., bidirectional=False):
        super().__init__()

        assert batch_first == True

        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.n_dirs = bidirectional + 1

    def reset_parameters(self):
        for m in self.modules():
            m.reset_parameters()

    def _init_hidden_state(self, inputs):
        raise NotImplementedError

    def forward(self, inputs, hidden_state=None):
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
    def __init__(self, mode, input_size, hidden_size, bias=True):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size

        if mode == 'LSTM':
            self.n_weights = 4
        elif mode == 'GRU':
            self.n_weights = 3


        self.linear_ih = LinearVD(input_size, hidden_size, bias=bias)
        self.linear_hh = LinearVD(hidden_size, hidden_size, bias=bias)

    def reset_parameters(self):
        # stdv = 1.0 / math.sqrt(self.hidden_size)
        for name, weight in self.named_parameters():
            if 'linear_hh.bias' in name:
                if self.mode == 'GRU':
                    self.bias_hh.data[:self.hidden_size] = 1
                elif self.mode == 'LSTM':
                    self.bias_hh[self.hidden_size:self.hidden_size * 2] = 1
            # else:
            #     torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, hidden_state):
        raise NotImplementedError


class RNNLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args, **cell_kwargs):
        super().__init__()
        self.cell = cell(*cell_args, **cell_kwargs)

    def forward(self, inputs, hidden_state):
        outputs = []
        for i in range(len(inputs)):
            out, hidden_state = self.cell(inputs[i], hidden_state)
            outputs += [out]
        return torch.stack(outputs), hidden_state


class ReverseRNNLayer(RNNLayer):
    def forward(self, inputs, hidden_state):
        outputs = []
        for i in reversed(range(len(inputs))):
            out, hidden_state = self.cell(inputs[i], hidden_state)
            outputs += [out]
        return torch.stack(outputs), hidden_state


class BidirRNNLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args, **cell_kwargs):
        super().__init__()
        self.cells = torch.nn.ModuleList([
            RNNLayer(cell, *cell_args, **cell_kwargs),
            ReverseRNNLayer(cell, *cell_args, **cell_kwargs)
        ])

    def forward(self, inputs, hidden_state):
        outputs = []
        output_hidden_states = torch.empty_like(hidden_state)

        for i, cell in enumerate(self.cells):
            state = hidden_state[i]
            out, out_state = cell(inputs, state)
            outputs += [out]
            output_hidden_states[i] = out_state

        return torch.cat(outputs, -1), output_hidden_states


class GRUCell(RNNCell):
    def __init__(self, input_size, hidden_size):
        super().__init__('GRU', input_size, hidden_size)

    def forward(self, input, hidden_state):

        # update dropout noise
        if self.training:
            self.linear_ih.update_weight_noise()
            self.linear_hh.update_weight_noise()

        gates_input = self.linear_ih(input)
        gates_hidden = self.linear_hh(hidden_state)

        rx, zx, nx = gates_input.chunk(self.n_weights, 1)
        rh, zh, nh = gates_hidden.chunk(self.n_weights, 1)

        r = F.sigmoid(rx + rh)
        z = F.sigmoid(zx + zh)
        n = F.tanh(nx + r * nh)

        hidden_state = (1 - z) * n + z * hidden_state
        return hidden_state, hidden_state

class GRU(RNNBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cell_args = (self.input_size, self.hidden_size, self.bias)
        layer = BidirRNNLayer if self.bidirectional else RNNLayer
        self.layers = torch.nn.ModuleList([
            layer(GRUCell, *cell_args) for _ in range(self.num_layers)
        ])
        self.n_dirs = self.bidirectional + 1

    def _init_hidden_state(self, inputs):
        batch_size = len(inputs)
        hidden_state = torch.zeros(
                self.num_layers, self.n_dirs, batch_size, self.hidden_size,
                dtype=inputs.dtype, device=inputs.device
        )
        return hidden_state

    def forward(self, inputs, hidden_state=None):
        if hidden_state is None:
            hidden_state = self._init_hidden_state(inputs)

        output_states = torch.empty_like(hidden_state)
        output = inputs

        for i, rnn_layer in enumerate(self.layers):
            state = hidden_state[i]
            output, out_state = rnn_layer(output, state)
            output_states[i] = out_state

        return output, output_states
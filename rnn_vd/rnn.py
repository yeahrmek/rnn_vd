import math
import torch

from cplxmodule.nn.masked import LinearMasked

from .vardropout import LinearVD


class BaseRNN(torch.nn.Module):
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

        first_cell_args = (input_size, hidden_size, bias, activation)
        other_cell_args = (hidden_size * self.n_dirs, hidden_size, bias,
                           activation)
        
        layer = BidirRNNLayer if bidirectional else RNNLayer
        self.layers = torch.nn.ModuleList(
                [layer(cell, *first_cell_args, **kwargs)] +
                [layer(cell, *other_cell_args, **kwargs)
             for _ in range(num_layers - 1)]
        )

    def reset_parameters(self):
        for m in self.modules():
            m.reset_parameters()


class BaseCell(torch.nn.Module):
    def __init__(self, mode, input_size, hidden_size, bias, linear,
                 activation, layernorm=False, **kwargs):
        super().__init__()
        self.mode = mode
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layernorm = layernorm
        
        if mode == 'LSTM':
            self.n_weights = 4
        elif mode == 'GRU':
            self.n_weights = 3

        self.linear_ih = linear(
                input_size, self.n_weights * hidden_size, bias=bias)
        self.linear_hh = linear(
                hidden_size, self.n_weights * hidden_size, bias=bias)
        self.activation = activation(**kwargs)

        if layernorm:
            self.layernorm_ih = torch.nn.LayerNorm(self.n_weights * hidden_size)
            self.layernorm_hh = torch.nn.LayerNorm(self.n_weights * hidden_size)

        self.reset_parameters()

    def forward(self, *args, **kwargs):
        pass

    def reset_parameters(self):
        for name, weight in self.named_parameters():
            if 'linear_hh.bias' in name:
                if self.mode == 'GRU':
                    weight.data[:self.hidden_size] = 1
                elif self.mode == 'LSTM':
                    weight.data[self.hidden_size:self.hidden_size * 2] = 1


class GRUCellMixin(object):
    def forward(self, input, hidden_state):
        gates_input = self.linear_ih(input)
        gates_hidden = self.linear_hh(hidden_state)
        
        if self.layernorm:
            gates_input = self.layernorm_ih(gates_input)
            gates_hidden = self.layernorm_ih(gates_hidden)
        
        rx, zx, nx = gates_input.chunk(self.n_weights, 1)
        rh, zh, nh = gates_hidden.chunk(self.n_weights, 1)
        
        r = torch.sigmoid(rx + rh)
        z = torch.sigmoid(zx + zh)
        n = self.activation(nx + r * nh)
        
        hidden_state = (hidden_state - n) * z + n
        return hidden_state, hidden_state
        

class GRUCell(GRUCellMixin, BaseCell):
    def __init__(self, input_size, hidden_size, bias,
                 activation=torch.nn.Tanh, **kwargs):
        super().__init__('GRU', input_size, hidden_size, bias, torch.nn.Linear,
                         activation, **kwargs)


class GRUVDCell(GRUCellMixin, BaseCell):
    def __init__(self, input_size, hidden_size, bias,
                 activation=torch.nn.Tanh, **kwargs):
        super().__init__('GRU', input_size, hidden_size, bias, LinearVD,
                         activation, **kwargs)

    def forward(self, input, hidden_state, **kwargs):
        # update dropout noise
        if self.training:
            self.linear_ih.update_weight_noise()
            self.linear_hh.update_weight_noise()

        return super().forward(input, hidden_state, **kwargs)


class GRUMaskedCell(GRUCellMixin, BaseCell):
    def __init__(self, input_size, hidden_size, bias, activation=torch.nn.Tanh,
                 **kwargs):
        super().__init__('GRU', input_size, hidden_size, bias, LinearMasked,
                         activation, **kwargs)


class RNNLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args, **cell_kwargs):
        super().__init__()
        self.cell = cell(*cell_args, **cell_kwargs)

    def forward(self, inputs, hidden_state, **kwargs):
        hidden_state = hidden_state.reshape(-1, self.cell.hidden_size)
        seq_len = inputs.shape[1]
    
        outputs = []
        for i in range(seq_len):
            out, hidden_state = self.cell(inputs[:, i], hidden_state)
            outputs += [out]
        return torch.stack(outputs, dim=1), hidden_state


class ReverseRNNLayer(RNNLayer):
    def forward(self, inputs, hidden_state, chunk_len=None, chunk_stride=None):
        """
        Forward pass of ReverseRNN layer. The sequence in split into small
        sequences of `chunk_len` size, for each such sequence the
        hidden state is reset to zeros (except the first one).
        The sequences are enumerate from the tail of the original sequence
        
        Args:
            inputs : torch.Tensor, shape=(batch, seq_len, features)
            
            hidden_state:
            
            chunk_len : int or None
                Length of sequence to split the original sequence to

        Returns:

        """
        hidden_state = hidden_state.reshape(-1, self.cell.hidden_size)
        seq_len = inputs.shape[1]
    
        if chunk_len is None:
            chunk_len = seq_len
        if chunk_stride is None:
            chunk_stride = chunk_len
            
        n_chunks = int(math.ceil(seq_len / chunk_stride))
    
        outputs = []
        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_stride
            end = min((chunk_idx + 1) * chunk_len, seq_len)

            idx_to_append = list(range(start, start + chunk_stride))
            if chunk_idx == n_chunks - 1:
                idx_to_append = list(range(start, end))
            
            hidden_state = torch.zeros_like(hidden_state)
            
            for i in range(end - 1, start - 1, -1):
                out, hidden_state = self.cell(inputs[:, i], hidden_state)
                if i in idx_to_append:
                    outputs += [out]
        
        return torch.stack(outputs[::-1], dim=1), hidden_state


class BidirRNNLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args, **cell_kwargs):
        super().__init__()
        self.straight = RNNLayer(cell, *cell_args, **cell_kwargs)
        self.reverse = ReverseRNNLayer(cell, *cell_args, **cell_kwargs)

    def forward(self, inputs, hidden_state, chunk_len, chunk_stride):
        hidden_state = hidden_state.reshape(
            2, -1, self.straight.cell.hidden_size)
    
        output_hidden_states = torch.empty_like(hidden_state)
    
        state = hidden_state[0]
        out, output_hidden_states[0] = self.straight(inputs, state)
        outputs = [out]
    
        state = hidden_state[1]
        out, output_hidden_states[1] = self.reverse(inputs, state, chunk_len,
                                                    chunk_stride)
        outputs += [out]
    
        return torch.cat(outputs, -1), output_hidden_states


class GRUMixin(object):
    def _init_hidden_state(self, inputs):
        batch_size = len(inputs)
        hidden_state = torch.zeros(
                self.num_layers * self.n_dirs, batch_size, self.hidden_size,
                dtype=inputs.dtype, device=inputs.device
        )

        return hidden_state

    def forward(self, inputs, hidden_state=None, **kwargs):
        if hidden_state is None:
            hidden_state = self._init_hidden_state(inputs)

        hidden_state = hidden_state.reshape(self.num_layers, self.n_dirs,
                                            -1, self.hidden_size)

        output_states = torch.empty_like(hidden_state)
        output = inputs

        for i, rnn_layer in enumerate(self.layers):
            state = hidden_state[i]
            output, out_state = rnn_layer(output, state, **kwargs)
            output_states[i] = out_state

        return output, output_states.reshape(self.num_layers * self.n_dirs,
                                             -1, self.hidden_size)

    def from_module(self, rnn_module):
        """
        Transfer weights from GRU/LSTM model

        Args:
            rnn_module : torch.nn.Module
                GRU model to transfer weights from

        Returns:

        """
        mapping = {}
        if self.bidirectional:
            name_format = "layers.{}.{}.cell.linear_{}.{}"
        else:
            name_format = "layers.{}.cell.linear_{}.{}"

        for name, _ in rnn_module.named_parameters():
            layer_number = name.split('_l')[-1].split('_reverse')[0]

            format_args = [
                layer_number,
                'straight' if not 'reverse' in name else 'reverse',
                'ih' if '_ih' in name else 'hh',
                'weight' if 'weight' in name else 'bias'
            ]
            if not self.bidirectional:
                format_args.pop(1)


            gru_vd_param_name = name_format.format(*format_args)

            mapping[name] = gru_vd_param_name

        state_dict = rnn_module.state_dict()
        state_dict = dict(
                (mapping[key], value) for key, value in state_dict.items()
        )

        return self.load_state_dict(state_dict, strict=False)


class GRU(GRUMixin, BaseRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(GRUCell, *args, **kwargs)


class GRUVD(GRUMixin, BaseRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(GRUVDCell, *args, **kwargs)


class GRUMasked(GRUMixin, BaseRNN):
    def __init__(self, *args, **kwargs):
        super().__init__(GRUMaskedCell, *args, **kwargs)

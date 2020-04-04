import math

import torch
import torch.nn.functional as F


class Attention(torch.nn.Module):
    """
    Base class for attention mechanism

    Parameters
    ----------
    input_size : int
        Size of the input vectors (aka hidden_size of the encoder)

    target_size : int
        Size of the target vectors (aka hidden_size of the decoder)

    score : torch.nn.Module
        Module that calculates similarity between input and target vectors
        It should take 2 arguments:

            inputs, shape=(batch, input_seq_len, input_size)
            targets, shape=(batch, target_seq_len, target_size)

        and return attention scores of shape (batch_size, output_seq_len,
        input_seq_len)

    """
    def __init__(self, score):
        super().__init__()

        self.score = score

    def forward(self, inputs, targets):
        """

        Args:

            inputs : torch.Tensor, shape=(batch, input_seq_len, input_size)
                Input sequence

            targets : torch.Tensor, shape=(batch, target_seq_len, target_size)
                Target sequence

        Returns:

        """
        attention_scores = self.score(inputs, targets)
        attention_weights = F.softmax(attention_scores, dim=-1)
        context_vector = attention_weights.unsqueeze(-2) @ inputs.unsqueeze(-3)

        return context_vector.squeeze(-2), attention_weights


class DotProductScore(torch.nn.Module):
    """
    Dot product attention score:

    \[
        score_i = inputs_i^\top targets_i
    \]

    """
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, inputs, targets):
        score = targets @ inputs.transpose(-2, -1)
        return score


class ScaledDotProductScore(DotProductScore):
    """
    Scaled dot product attention score:

    \[
        score_i = inputs_i^\top targets_i / \sqrt{n},
    \]
    where `n` is a sequence length

    """
    def forward(self, inputs, targets):
        seq_len = inputs.shape[1]
        score = super().forward(inputs, targets)
        return score / math.sqrt(seq_len)


class GeneralDotProductScore(DotProductScore):
    def __init__(self, input_size, target_size, linear=torch.nn.Linear):
        super().__init__()
        self.linear = linear(target_size, input_size, bias=False)

    def forward(self, inputs, targets):
        targets = self.linear(targets)
        score = super().forward(inputs, targets)
        return score


class TransformerEncoderLayer(torch.nn.Module):
    r"""This is a fork of the `torch.nn.TransformerEncoderLayer` with
    additional parameters `kdim` and `vdim`.
    """

    def __init__(self, d_model, nhead, kdim=None, vdim=None,
                 dim_feedforward=2048, dropout=0.1, activation="relu",
                 linear=torch.nn.Linear):
        super().__init__()
        self.self_attn = torch.nn.MultiheadAttention(
            d_model, nhead, kdim=kdim, vdim=vdim, dropout=dropout)
        
        # Implementation of Feedforward model
        self.linear1 = linear(d_model, dim_feedforward)
        self.dropout = torch.nn.Dropout(dropout)
        self.linear2 = linear(dim_feedforward, d_model)

        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
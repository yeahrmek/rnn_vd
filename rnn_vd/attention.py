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



import torch
import torch.nn.functional as F
from cplxmodule.relevance import LinearARD


class LinearVD(LinearARD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('weight_noisy', torch.zeros_like(self.weight))

    def forward(self, input):

        if self.training:
            weight = self.weight_noisy
        else:
            weight = self.weight

        return F.linear(input, weight, bias=self.bias)

    def update_weight_noise(self):
        noise = torch.randn_like(self.weight)

        std = torch.clamp(torch.exp(self.log_sigma2 / 2), min=1e-8)

        self.weight_noisy = self.weight + noise * std
import math

import torch
from torch import nn


class LowRankLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.weight_u = nn.Parameter(
            torch.empty(rank, in_dim),
            requires_grad=True
        )
        self.weight_v = nn.Parameter(
            torch.empty(out_dim, rank),
            requires_grad=True
        )
        self.bias = nn.Parameter(
            torch.empty(out_dim),
            requires_grad=True
        )

        var = 6.0 / (in_dim + out_dim)  # replace 6.0 with 2.0 for normal distribution init
        var /= rank
        var *= 2  # gain for ReLU activation
        nn.init.uniform_(self.weight_u, -math.sqrt(var), math.sqrt(var))
        nn.init.uniform_(self.weight_v, -math.sqrt(var), math.sqrt(var))

    def forward(self, input):
        # import pdb; pdb.set_trace()
        return torch.linalg.multi_dot([self.weight_v, self.weight_u, input.t()]).t() + self.bias

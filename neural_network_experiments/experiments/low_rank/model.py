from neural_network_experiments.lib import low_rank_layer as lrl
from torch import nn


class LowRankNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dim: int, rank: int):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            lrl.LowRankLinear(input_dim, dim, rank=rank),
            nn.ReLU(),
            lrl.LowRankLinear(dim, dim, rank=rank),
            nn.ReLU(),
            lrl.LowRankLinear(dim, output_dim, rank=rank),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)

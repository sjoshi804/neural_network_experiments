from torch import functional as F
from torch import nn 
import torch
from neural_network_experiments.lib.util import outer_product_tensor_w_vector as prod

class LowRankConv2D(nn.Module):
    def __init__(
        self,
        rank,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple(int, int),
        padding: int = 0,
    ):
        super(LowRankConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.rank = rank

        # Creating parameters
        self.dim1 = nn.ParameterList()
        self.dim2 = nn.ParameterList()
        self.dim3 = nn.ParameterList()
        self.dim4 = nn.ParameterList()
        for i in range(rank):
            self.dim1.append(torch.rand(in_channels))
            self.dim2.append(torch.rand(out_channels))
            self.dim3.append(torch.rand(kernel_size[0]))
            self.dim4.append(torch.rand(kernel_size[1]))

    def forward(self, x):
        weights = None
        for i in range(self.rank):
            product = prod(prod(prod(self.dim1[0], self.dim2[0]), self.dim3[0]), self.dim4[0])
            if weights is None:
                weights = product 
            else: 
                weights += product
        return F.conv2d(x, weights, padding=self.padding)
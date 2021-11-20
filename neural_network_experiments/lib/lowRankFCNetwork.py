from torch import nn 
from neural_network_experiments.lib.low_rank_layer import LowRankLinear

class LowRankFCNetwork(nn.Module):
    def __init__(self, layers_dim, layers_rank) -> None:
        '''
        layers_dim: List with # of neurons in each layer - starting with input and ending with output 
        layers_rank: constraint on rank for low rank layer
        '''
        super().__init__()

        # Creating the model
        self.flatten = nn.Flatten()
        self.layers = []
        for i in range(len(layers_dim) - 1):
            self.layers.append(LowRankLinear(layers_dim[i], layers_dim[i+1], min(layers_dim[i], layers_dim[i+1], layers_rank[i])))
            if i + 1 < (len(layers_dim) - 1):
                self.layers.append(nn.ReLU())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)



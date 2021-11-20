from torch import nn

class  FCNetwork(nn.Module):
    def __init__(self, layers_dim, expansion_factor=1) -> None:
        '''
        layers_dim: List with # of neurons in each layer - starting with input and ending with output
        '''
        super().__init__()

        # Creating the model
        self.flatten = nn.Flatten()
        self.layers = []
        for i in range(len(layers_dim) - 1):
            self.layers.append(nn.Linear(layers_dim[i], layers_dim[i+1]))
            for _ in range(1, expansion_factor):
                self.layers.append(nn.Linear(layers_dim[i+1], layers_dim[i+1]))
            if i + 1 < (len(layers_dim) - 1):
                self.layers.append(nn.ReLU())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)




        
from torch import nn

class  FCNetwork(nn.Module):
    def __init__(self, layers_dim) -> None:
        '''
        layers_dim: List with # of neurons in each layer - starting with input and ending with output
        '''
        super().__init__()

        # Creating the model
        self.layers = []
        for i in range(len(layers_dim) - 1):
            self.layers.append(nn.Linear(layers_dim[i], layers_dim[i+1]))
            self.layers.append(nn.ReLU())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)




        
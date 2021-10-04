from torch import nn

class FCNetwork(nn.Module):
    def __init__(self, layers_dim) -> None:
        '''
        layers_dim: List with # of neurons in each layer - starting with input and ending with output
        '''
        super().__init__()

        # Validate Input
        assert(len(layers_dim) >= 3) # atleast one hidden layer

        # Creating the model
        self.model = nn.ModuleList()
        for i in range(len(layers_dim) - 1):
            self.model.append(nn.Linear(layers_dim[i], layers_dim[i+1]))

    def forward(self, x):
        return self.model(x)





        
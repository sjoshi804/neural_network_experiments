from lib.fcNetwork import FCNetwork
import torch

class SyntheticData():
    def __init__(self, num_samples, layers_dim) -> None:   
        '''
        layers_dim: layers_dim for Teacher Model - fully connected network that determines the labels
        '''
        # Create Teacher Model
        self.teacher = FCNetwork(layers_dim=layers_dim)

        self.data = []
        
        for i in range(num_samples):
            # X ~ normal distribution
            x = torch.normal(torch.tensor([0] * layers_dim[0]), torch.tensor([1] * layers_dim[0]))
            # Y = Teacher(x)
            y = self.teacher(x)
            # Noise ~ normal distribution with small variance
            noise = torch.normal(torch.tensor([0] * layers_dim[0]), torch.tensor([0.01] * layers_dim[0]))
            # Data = (X, y + Noise)
            self.data.append((x, y + noise))
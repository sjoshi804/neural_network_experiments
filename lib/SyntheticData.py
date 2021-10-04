from lib.FCNetwork import FCNetwork
import random
from torch import tensor 

class SyntheticData():
    def __init__(self, num_samples, layers_dim) -> None:   
        '''
        layers_dim: layers_dim for Teacher Model - fully connected network that determines the labels
        '''
        # Create Teacher Model
        self.teacher = FCNetwork(layers_dim=layers_dim)

        self.data = []
        
        for i in range(num_samples):
            x = tensor([random.random()] * layers_dim[0]) # Sample from normal distribution
            y = self.teacher(x)
            noise = tensor([random.random() * 0.1] * layers_dim[-1])# sample some noise from normal distribution
            self.data.append((x, y + noise))
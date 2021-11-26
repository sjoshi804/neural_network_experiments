from torch import nn, reshape

class ModelFromPaper(nn.Module):
    def __init__(self, expansion_factor):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), 
            nn.MaxPool2d(2),
            nn.ReLU())

        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), 
            nn.MaxPool2d(2),
            nn.ReLU())

        self.cnn3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), 
            nn.MaxPool2d(2),
            nn.ReLU())

        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), 
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

        self.linear1 = self.expanded_linear_layer(512, 256, expansion_factor)
        self.linear1_activation = nn.ReLU()
        self.linear2 = nn.Sequential(self.expanded_linear_layer(256, 10, expansion_factor))
        self.linear2_activation = nn.ReLU()

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = reshape(x, [x.shape[0], x.shape[1]])
        x = self.linear1_activation(self.linear1(x))
        return self.linear2_activation(self.linear2(x))
    
    def expanded_linear_layer(self, in_dim, out_dim, expansion_factor):
        layers = [nn.Linear(in_dim, out_dim)]
        layers.extend([nn.Linear(out_dim, out_dim)] * (expansion_factor - 1))
        return nn.Sequential(*layers)
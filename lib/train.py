from torch import optim
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Train():
    def __init__(self, data, model, loss_fn) -> None:
        self.data = data.to(device)
        self.model = model.to(device) 
        self.loss_fn = loss_fn
        self.optimizer = optim.SGD(model.parameters())
        self.losses = []

    def train_to_convergence(self, convergence_threshold, max_iterations):
        loss = float('inf')
        i = 0
        while loss > convergence_threshold and i < max_iterations:
            loss = self.train()
            self.losses.append(loss)
            i += 1

    def train(self):
        for input, target in self.data:
            self.optimizer.zero_grad()
            output = self.model(input)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
        return loss
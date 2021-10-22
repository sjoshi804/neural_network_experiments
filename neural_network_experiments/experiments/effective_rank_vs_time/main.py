import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from neural_network_experiments.lib.fcNetwork import FCNetwork
from neural_network_experiments.lib.util import get_data, effective_rank

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.reshape(-1, 28*28).to(device)
        pred = model(X)
        # import pdb; pdb.set_trace()
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.reshape(-1, 28*28).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def main(device: str):
    training_data, test_data = get_data("MNIST")
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    input_dim = int(np.prod(training_data.data.shape[1:]))
    output_dim = len(training_data.classes)

    model = FCNetwork([input_dim, 100, 100, output_dim]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 1

    ranks = []
    for layer in model.layers:
        if type(layer) is not nn.ReLU:
            ranks.append([])

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)

        # Compute effective rank for each layer
        i = 0
        for layer in model.layers:
            if type(layer) is not nn.ReLU:
                ranks[i].append(effective_rank(layer.state_dict()['weight']))
                i += 1

    print(ranks)
    print("Done!")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    main(device)

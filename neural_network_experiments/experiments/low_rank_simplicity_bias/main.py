import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from neural_network_experiments.lib.fcNetwork import FCNetwork
from neural_network_experiments.lib.lowRankFCNetwork import LowRankFCNetwork
from neural_network_experiments.lib.util import get_data, effective_rank
import matplotlib.pyplot as plt

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        # import pdb; pdb.set_trace()
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss calc
        train_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    return train_loss / len(dataloader) 

def test_loop(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (100*correct)

def train_and_test(model, train_dataloader, test_dataloader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    test_accuracies = []

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_losses.append(train_loop(train_dataloader, model, loss_fn, optimizer, device))
        test_accuracies.append(test_loop(test_dataloader, model, loss_fn, device))
    
    return train_losses, test_accuracies

def main(device: str):
    # Get data loaders
    training_data, test_data = get_data("CIFAR10")
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    epochs = 50

    # Determine input, output dim
    input_dim = int(np.prod(training_data.data.shape[1:]))
    output_dim = len(training_data.classes)

    # Dimensionality of model
    layers_dim = [input_dim, 100, 100, 100, output_dim]

    losses = []
    accuracies = []
    for multiplier in [1, 2, 4, 8]:
        model = FCNetwork(layers_dim, multiplier).to(device)
        loss, accuracy = train_and_test(model, train_dataloader, test_dataloader, epochs)
        losses.append(loss)
        accuracies.append(accuracy)

    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.title("Investigating Low Rank Simplicity Bias")
    for i, accuracy in enumerate(accuracies):
        plt.plot(range(epochs), accuracy, label=str(2**(i-1)))
    
    plt.savefig("low_rank_simplicity_bias")

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    main(device)

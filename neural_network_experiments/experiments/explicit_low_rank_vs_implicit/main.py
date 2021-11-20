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
    epochs = 2

    # Determine input, output dim
    input_dim = int(np.prod(training_data.data.shape[1:]))
    output_dim = len(training_data.classes)

    # Dimensionality of model
    layers_dim = [input_dim, 10, output_dim]

    # Create implicit low rank model
    implicit_low_rank_model = FCNetwork(layers_dim, 4).to(device)

    # Train and test model
    implicit_low_rank_train_losses, implicit_low_rank_test_accuracies = train_and_test(implicit_low_rank_model, train_dataloader, test_dataloader, epochs)

    # Create explicit low rank model
    explicit_low_rank_model = LowRankFCNetwork(layers_dim, [10, 10]).to(device)

    # Train and test model
    explicit_low_rank_train_losses, explicit_low_rank_test_accuracies = train_and_test(explicit_low_rank_model, train_dataloader, test_dataloader, epochs)

    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.plot(epochs, implicit_low_rank_test_accuracies, label="Implicit Low Rank")
    plt.plot(epochs, explicit_low_rank_test_accuracies, label="Explicit Low Rank")
    plt.show()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    main(device)

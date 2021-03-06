import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from neural_network_experiments.lib.util import get_data, effective_rank
from neural_network_experiments.lib.low_rank_layer import LowRankLinear
from modelfrompaper import ModelFromPaper, ModelFromPaperWithLowRankLayer
import matplotlib.pyplot as plt

def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Loss calc
        train_loss += loss.item()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
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
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return (100*correct)

def train_and_test(model, train_dataloader, test_dataloader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    test_accuracies = []
    effective_ranks_1 = []
    effective_ranks_2 = []

    for t in range(epochs):
        # print(f"Epoch {t+1}\n-------------------------------")
        train_losses.append(train_loop(train_dataloader, model, loss_fn, optimizer, device))
        test_accuracies.append(test_loop(test_dataloader, model, loss_fn, device))
        effective_ranks_1.append(get_effective_rank(model.linear1))
        effective_ranks_2.append(get_effective_rank(model.linear2))

    return train_losses, test_accuracies, effective_ranks_1, effective_ranks_2

def get_effective_rank(linear_layers):
    if type(linear_layers) is LowRankLinear:
        matrix = torch.linalg.multi_dot([linear_layers.weight_v.cpu(), linear_layers.weight_u.cpu()])
    else:  
        matrix = None
        for layer in linear_layers.modules():

            if type(layer) is not nn.Linear:
                continue

            elif matrix is None:
                matrix = layer.weight.cpu()
            else:
                matrix = torch.matmul(layer.weight.cpu(), matrix)
    
    return effective_rank(matrix)

def drop_rank(model, new_rank):
    model.linear1 = LowRankLinear(512, 256, new_rank)
def main(device: str):
    # Get data loaders
    training_data, test_data = get_data("CIFAR10")
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    epochs = 100

    '''
    # Determine input, output dim
    input_dim = int(np.prod(training_data.data.shape[1:]))
    output_dim = len(training_data.classes)
    '''

    losses = []
    accuracies = []
    effective_ranks_1 = []
    effective_ranks_2 = []
    labels = []

    # Drop at epoch 0 
    rank1 = 160
    model = ModelFromPaperWithLowRankLayer(rank1).to(device)
    loss, accuracy, effective_rank_1, effective_rank_2 = train_and_test(model, train_dataloader, test_dataloader, epochs)
    losses.append(loss)
    accuracies.append(accuracy)
    effective_ranks_1.append(effective_rank_1)
    effective_ranks_2.append(effective_rank_2)
    labels.append("Drop rank at init")

    # Drop at epoch 1
    model = ModelFromPaperWithLowRankLayer(rank1).to(device)
    loss, accuracy, effective_rank_1, effective_rank_2 = train_and_test(model, train_dataloader, test_dataloader, 1)
    model = drop_rank(model, 160)
    loss_2, accuracy_2, effective_rank_1_2, effective_rank_2_2 = train_and_test(model, train_dataloader, test_dataloader, epochs-1)
    loss.extend(loss_2)
    accuracy.extend(accuracy_2)
    effective_rank_1.extend(effective_rank_1_2)
    effective_rank_2_2.extend(effective_rank_2_2)
    losses.append(loss)
    accuracies.append(accuracy)
    effective_ranks_1.append(effective_rank_1)
    effective_ranks_2.append(effective_rank_2)
    labels.append("Drop rank at epoch 1")

    # Drop at epoch 1
    model = ModelFromPaperWithLowRankLayer(rank1).to(device)
    loss, accuracy, effective_rank_1, effective_rank_2 = train_and_test(model, train_dataloader, test_dataloader, 1)
    model = drop_rank(model, 160)
    loss_2, accuracy_2, effective_rank_1_2, effective_rank_2_2 = train_and_test(model, train_dataloader, test_dataloader, epochs-1)
    loss.extend(loss_2)
    accuracy.extend(accuracy_2)
    effective_rank_1.extend(effective_rank_1_2)
    effective_rank_2_2.extend(effective_rank_2_2)
    losses.append(loss)
    accuracies.append(accuracy)
    effective_ranks_1.append(effective_rank_1)
    effective_ranks_2.append(effective_rank_2)
    labels.append("Drop rank at epoch 2")



    plt.xlabel("Epochs")
    plt.ylabel("Test Accuracy")
    plt.title("Investigating Low Rank Simplicity Bias: Accuracy")
    
    for i, accuracy in enumerate(accuracies):
        plt.plot(range(epochs), accuracy, label=labels[i])
    plt.legend()
    plt.savefig("low_rank_simplicity_bias_acc")
    plt.clf()

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Investigating Low Rank Simplicity Bias: Loss")
    for i, loss in enumerate(losses):
        plt.plot(range(epochs), loss, label=labels[i])
    plt.legend()
    plt.savefig("low_rank_simplicity_bias_loss")
    plt.clf()

    plt.xlabel("Epochs")
    plt.ylabel("Effective Rank")
    plt.title("Investigating Low Rank Simplicity Bias: Eff Rank Layer 1")
    for i, rank in enumerate(effective_ranks_1):
        rs = [r.detach().numpy() for r in rank]
        plt.plot(range(epochs), rs, label=labels[i])
    plt.legend()
    plt.savefig("low_rank_simplicity_bias_eff_rank_1")
    plt.clf()

    plt.xlabel("Epochs")
    plt.ylabel("Effective Rank")
    plt.title("Investigating Low Rank Simplicity Bias: Eff Rank Layer 2")
    for i, rank in enumerate(effective_ranks_2):
        rs = [r.detach().numpy() for r in rank]
        plt.plot(range(epochs), rs, label=labels[i])
    plt.legend()
    plt.savefig("low_rank_simplicity_bias_eff_rank_2")
    plt.clf()

    print("Losses")
    print(losses)
    print("Accuracies")
    print(accuracies)
    print("Effective Rank 1")
    print(effective_ranks_1)
    print("Effective Rank 2")
    print(effective_ranks_2)

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    main(device)

from torch import tensor
import numpy as np 
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

def get_data(dataset: str):
    if dataset == "CIFAR10":
        training_data = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
            # target_transform=Lambda(
            #     lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            # )
        )
        test_data = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
            # target_transform=Lambda(
            #     lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            # )
        )
        return training_data, test_data

    elif dataset == "MNIST":
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor(),
            # target_transform=Lambda(
            #     lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            # )
        )
        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor(),
            # target_transform=Lambda(
            #     lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)
            # )
        )
        return training_data, test_data
    else:
        raise Exception("dataset not supported")

def split_data_into_batches(data, batch_size):
    batched_data = []
    x = []
    y = []
    for i in range(len(data)):
        if (i % batch_size == 0):
            batched_data.append((tensor(x), tensor(y)))
            x = []
            y = []
        x.append(data[i][0])
        y.append(data[i][1])
    return batched_data

def train_test_split(data, train_fraction=0.8):
    split_index = int(0.8 * len(data))
    train_data = data[:split_index]
    test_data = data[split_index:]
    return train_data, test_data

def test_loss(test_data, model, loss_fn):
    model.eval()
    losses = []
    for input, target in test_data:
        output = model(input)
        losses.append(loss_fn(output, target))
    return np.mean(losses)

def outer_product_tensor_w_vector(a_tensor: tensor, a_vector: tensor):
    # Assert that arguments are correct
    if len(a_vector.shape) != 1:
        raise Exception("Invalid argument: a_vector must be a 1d tensor")
    if len(a_tensor.shape) < 1:
        raise Exception("Invalid argument: a_tensor must not be a scalar")
    # Shape of final tensor will be a_vector.shape.extend(a_tensor.shape)
    result = torch.kron(a_vector, a_tensor)
    return result.reshape(a_vector.shape + a_tensor.shape)


def effective_rank(weight_matrix):
    _, singular_values, _ = torch.svd(weight_matrix)
    sum = torch.sum(singular_values)
    normalized_singular_values = singular_values/sum
    return entropy(normalized_singular_values)

def entropy(prob):
    log_prob = torch.log2(prob)
    return torch.sum(-prob*log_prob)/torch.log2(tensor(len(prob)))


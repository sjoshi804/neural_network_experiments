from torch import tensor
import numpy as np 

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
    split_index = 0.8 * len(data)
    train_data = data[split_index]
    test_data = data[split_index]
    return train_data, test_data

def test_loss(test_data, model, loss_fn):
    model.eval()
    losses = []
    for input, target in test_data:
        output = model(input)
        losses.append(loss_fn(output, target))
    return np.mean(losses)
import argparse
from scipy import stats
from typing import List

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, ReLU, Softmax
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

from low_rank_layer import LowRankDense


def train_rank(dataset: str, ranks: List[int]) -> pd.DataFrame:
    print(f"Starting training with ranks: {ranks}")
    if dataset == "mnist":
        load_data = mnist.load_data
    elif dataset == "cifar10":
        load_data = cifar10.load_data
    elif dataset == "cifar100":
        load_data = cifar100.load_data
    else:
        raise ValueError(f"Invalid dataset: {dataset}")
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.astype(np.float32) / np.max(x_train)
    x_test = x_test.astype(np.float32) / np.max(x_test)
    num_classes = np.max(y_train) + 1
    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    y_train = tf.one_hot(y_train, depth=num_classes).numpy()
    y_test = tf.one_hot(y_test, depth=num_classes).numpy()

    low_rank_model = Sequential([
        InputLayer(input_shape=x_train.shape[1:]),
        Flatten(),
        LowRankDense(200, init_rank=ranks[0]),
        ReLU(),
        LowRankDense(200, init_rank=ranks[1]),
        ReLU(),
        LowRankDense(200, init_rank=ranks[2]),
        ReLU(),
        LowRankDense(num_classes, init_rank=ranks[-1]),
        Softmax()
    ])
    low_rank_model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )

    results = []
    for epoch in range(30):
        print(f"Epoch {epoch}")
        result = {
            "epoch": epoch,
        }
        for i, r in enumerate(ranks):
            result[f"rank{i}"] = r

        train_hist = low_rank_model.fit(
            x_train,
            y_train,
            batch_size=128,
            validation_data=(x_test, y_test)).history
        result["acc"] = train_hist["categorical_accuracy"][-1]
        result["loss"] = train_hist["loss"][-1]
        result["val_acc"] = train_hist["val_categorical_accuracy"][-1]
        result["val_loss"] = train_hist["val_loss"][-1]

        low_rank_dense_ind = 0
        for i, layer in enumerate(low_rank_model.layers):
            if isinstance(layer, LowRankDense):
                try:
                    w, _ = layer.effective_weights()
                    _, svd, _ = np.linalg.svd(w, full_matrices=False)
                    svd /= np.sum(svd)
                    entr = stats.entropy(svd)
                except:
                    entr = float("nan")
                result[f"low_rank_{low_rank_dense_ind}_eff_rank"] = entr
                low_rank_dense_ind += 1

        assert low_rank_dense_ind == len(ranks), "number of provided ranks does not match number " \
                                                 "of low rank layers"

        results.append(result)
    return pd.DataFrame(results).set_index("epoch")


def main(dataset: str):
    hidden_layer_ranks = [20, 30, 40, 60, 80, 120, 140, 200]
    for rank1 in [80, 60, 40]:
        for rank2 in [80, 60, 40]:
            for rank3 in reversed(hidden_layer_ranks):
                for rank4 in [5, 10]:
                    result = train_rank(dataset, [rank1, rank2, rank3, rank4])
                    result.to_csv(f"./results/{dataset}_{rank1}_{rank2}_{rank3}_{rank4}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a low rank network and compares its "
                                                 "gradients with the effective model's gradients")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"])
    args = parser.parse_args()
    main(args.dataset)

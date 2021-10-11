import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.backend import count_params
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, InputLayer, ReLU, Softmax

from low_rank_layer import LowRankDense, Dense


def normalize_input(x: np.ndarray) -> np.ndarray:
    return x.astype("float32") / np.max(x)


def one_hot_label(y: np.ndarray) -> np.ndarray:
    y = np.squeeze(y)
    return tf.one_hot(y, np.max(y) + 1)


def get_model(input_shape: list[int], output_shape: int, dim: int, rank: int) -> Model:
    def get_dense():
        if rank == -1:
            return Dense(dim)
        else:
            return LowRankDense(dim, rank=rank)
    model = Sequential([
        InputLayer(input_shape=input_shape),
        Flatten(),
        get_dense(),
        # ReLU(),
        get_dense(),
        # ReLU(),
        Dense(output_shape),
        Softmax()
    ])
    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics="accuracy"
    )
    return model


def main(dataset="mnist"):
    if dataset == "mnist":
        load_data = mnist.load_data
    elif dataset == "cifar10":
        load_data = cifar10.load_data
    elif dataset == "cifar100":
        load_data = cifar100.load_data
    else:
        raise ValueError(f"Invalid dataset name: {dataset}")
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = normalize_input(x_train)
    y_train = one_hot_label(y_train)
    x_test = normalize_input(x_test)
    y_test = one_hot_label(y_test)

    results = []
    for dim in [10, 20, 50, 100, 200, 500]:
        for rank in [5, 10, 20, 50, 100, 200, 500, -1]:
            if rank > dim:
                continue
            model = get_model(x_train.shape[1:], y_train.shape[-1], dim=dim, rank=rank)
            num_params = np.sum([count_params(w) for w in model.trainable_weights])
            print(f"Dim: {dim}, Rank: {rank}, #Params: {num_params} - ", end="", flush=True)
            start_time = time.time()
            hist = model.fit(
                x_train,
                y_train,
                batch_size=256,
                validation_data=(x_test, y_test),
                epochs=50,
                verbose=0
            )
            end_time = time.time()
            training_duration = end_time - start_time
            print(f"Finished in {training_duration:.2f} seconds ", end="", flush=True)
            for epoch in range(len(hist.history["loss"])):
                epoch_res = {
                    "epoch": (epoch + 1),
                    "num_parameters": num_params,
                    "dim": dim,
                    "rank": rank,
                    "training_duration": training_duration
                }
                for metric, values in hist.history.items():
                    epoch_res[metric] = values[epoch]
                results.append(epoch_res)
            print(f"Final val acc: {hist.history['val_accuracy'][-1]:.2f}")
    results = pd.DataFrame(results)
    results.to_csv(f"results_linear_{dataset}.csv")
    print(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a low rank dense layer experiment")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mnist", "cifar10", "cifar100"]
    )
    args = parser.parse_args()
    main(dataset=args.dataset)

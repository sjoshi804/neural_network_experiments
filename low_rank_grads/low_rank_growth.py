import argparse
import random
from typing import List
from collections import defaultdict
import pprint

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
from low_rank_ops import create_eff_model, set_eff_weights, eff_gradients


def create_low_rank_model(ranks: List[int], input_shape, num_classes: int, name: str):
    low_rank_model = Sequential([
        InputLayer(input_shape=input_shape),
        Flatten(),
        LowRankDense(200, init_rank=ranks[0]),
        ReLU(),
        LowRankDense(200, init_rank=ranks[1]),
        ReLU(),
        LowRankDense(200, init_rank=ranks[2]),
        ReLU(),
        Dense(num_classes),
        Softmax()
    ], name=name)
    low_rank_model.compile(
        optimizer=RMSprop(learning_rate=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )
    return low_rank_model


def main(dataset: str):
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
    training_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    training_ds = training_ds.shuffle(buffer_size=1024).batch(128)

    low_rank_models = [
        # create_low_rank_model([80, 60, 40], x_train.shape[1:], num_classes, "80_60_40_low_rank"),
        # create_low_rank_model([80, 60, 20], x_train.shape[1:], num_classes, "80_60_20_low_rank"),
        # create_low_rank_model([80, 60, 10], x_train.shape[1:], num_classes, "80_60_10_low_rank"),
        # create_low_rank_model([40, 30, 10], x_train.shape[1:], num_classes, "40_30_10_low_rank"),
        create_low_rank_model([40, 30, 10], x_train.shape[1:], num_classes,
                              "40-80_30-60_10-40_low_rank"),
        create_low_rank_model([40, 30, 10], x_train.shape[1:], num_classes,
                              "40-80_30-60_10-40_low_rank2"),
        # create_low_rank_model([40, 20, 10], x_train.shape[1:], num_classes, "40_20_10_low_rank"),
        # create_low_rank_model([30, 20, 10], x_train.shape[1:], num_classes, "30_20_10_low_rank"),
        # create_low_rank_model([20, 20, 10], x_train.shape[1:], num_classes, "20_20_10_low_rank"),
    ]
    effective_model = create_eff_model(low_rank_models[0], "eff_model")
    effective_model.compile(
        optimizer=RMSprop(learning_rate=0.0001),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )

    models = low_rank_models #+ [effective_model]

    model_rank_additions = {
        "40-80_30-60_10-40_low_rank": [(2, [40, 30, 30])],
        "40-80_30-60_10-40_low_rank2": [(2, [20, 15, 15]), (4, [20, 15, 15])]
    }

    results = defaultdict(list)
    for epoch in range(200):
        print(f"Epoch {epoch}")
        for model in models:
            if model.name in model_rank_additions:
                for rank_add_config in model_rank_additions[model.name]:
                    if epoch == rank_add_config[0]:
                        rank_additions = rank_add_config[1]
                        rank_ind = 0
                        for layer in model.layers:
                            if isinstance(layer, LowRankDense):
                                layer.add_rank(rank_additions[rank_ind])
                                rank_ind += 1
            print(f"Training {model.name}")
            metrics = model.fit(
                training_ds,
                batch_size=64,
                epochs=1,
                validation_data=(x_test, y_test),
                verbose=2)
            result = {metric: val[0] for metric, val in metrics.history.items()}
            result["epoch"] = epoch
            results[model.name].append(result)
    for model_name, train_res in results.items():
        df = pd.DataFrame(train_res)
        df.to_csv(f"grow_train_results/{model_name}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a low rank network and compares its "
                                                 "gradients with the effective model's gradients")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"])
    args = parser.parse_args()
    main(args.dataset)

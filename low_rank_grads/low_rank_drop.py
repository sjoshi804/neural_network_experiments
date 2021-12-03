import argparse
import glob
import json
import random
from typing import List
from collections import defaultdict
import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense, ReLU, Softmax, Conv2D
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

from low_rank_layer import LowRankDense
from low_rank_ops import create_eff_model, set_eff_weights, eff_gradients


def main(dataset: str, run_name: str):
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

    model = Sequential([
        InputLayer(input_shape=x_train.shape[1:]),
        Conv2D(8, 3, activation="relu"),  # 64
        Conv2D(8, 3, activation="relu"),  # 64
        Flatten(),
        LowRankDense(256, 200),
        ReLU(),
        LowRankDense(256, 128),
        ReLU(),
        LowRankDense(256, 128),
        ReLU(),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(0.001),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )
    drop_config = json.load(open(f"drop_train_results/{run_name}.json"))
    print(drop_config)
    results = []
    epochs = 40
    for epoch in range(epochs):
        hist = model.fit(training_ds, validation_data=(x_test, y_test))
        low_rank_ind = 0
        for layer in model.layers:
            if isinstance(layer, LowRankDense):
                drop = drop_config[str(low_rank_ind)]
                if str(epoch) not in drop:
                    continue
                new_rank = drop[str(epoch)]
                print(f"Setting {low_rank_ind} to {new_rank}")
                layer.set_rank(new_rank)
                low_rank_ind += 1
        result = {"epoch": epoch}
        for k, v in hist.history.items():
            result[k] = v[0]
        results.append(result)
    results = pd.DataFrame(results)
    results.to_csv(f"drop_train_results/{run_name}.csv", index_label="epoch")


if __name__ == "__main__":
    runs = glob.glob("drop_train_results/*.json")
    runs.sort(
        key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[1]),  # run_x -> extract x
        reverse=True
    )
    runs = [x.split("/")[-1].split(".")[0] for x in runs]
    for run in runs:
        print(f"Starting {run}")
        main(dataset="cifar10", run_name=run)

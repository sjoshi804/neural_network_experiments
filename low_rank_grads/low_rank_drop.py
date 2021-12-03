import argparse
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
from tensorflow.keras.layers import InputLayer, Flatten, Dense, ReLU, Softmax
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy

from low_rank_layer import LowRankDense
from low_rank_ops import create_eff_model, set_eff_weights, eff_gradients


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

    model = Sequential([
        InputLayer(input_shape=x_train.shape[1:]),
        Flatten(),
        Dense(256, activation="relu"),
        Dense(256, activation="relu"),
        Dense(256, activation="relu"),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(0.001),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )
    results = []
    epochs = 40
    for epoch in range(epochs):
        result = {
            "epoch": epoch
        }
        layer_ind = 0
        for layer in model.layers:
            if isinstance(layer, Dense):
                w, b = layer.get_weights()
                _, s, _ = np.linalg.svd(w, full_matrices=False)
                result[f"layer{layer_ind}"] = s
                layer_ind += 1
        results.append(result)
        model.fit(training_ds, validation_data=(x_test, y_test))

    fig, axes = plt.subplots(4, epochs, figsize=(epochs*4, 4*6))
    for layer_ind in range(4):
        for epoch in range(epochs):
            ax = axes[layer_ind, epoch]
            s = results[epoch][f"layer{layer_ind}"]
            ax.hist(s, bins=min(50, len(s)))
            ax.set_title(f"layer {layer_ind}, epoch {epoch}")
    plt.show()
    fig.savefig("svd_over_epochs.png")


if __name__ == "__main__":
    main(dataset="cifar10")

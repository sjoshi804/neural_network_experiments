import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10, cifar100


def normalize_input(x: np.ndarray) -> np.ndarray:
    return x.astype("float32") / np.max(x)


def one_hot_label(y: np.ndarray) -> np.ndarray:
    y = np.squeeze(y)
    return tf.one_hot(y, np.max(y) + 1)


def load_data(dataset: str):
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
    return (x_train, y_train), (x_test, y_test)

import argparse
import random

import numpy as np

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

    low_rank_model = Sequential([
        InputLayer(input_shape=x_train.shape[1:]),
        Flatten(),
        LowRankDense(20, init_rank=10),
        ReLU(),
        LowRankDense(30, init_rank=10),
        ReLU(),
        LowRankDense(40, init_rank=10),
        ReLU(),
        LowRankDense(num_classes, init_rank=5),
        Softmax()
    ])
    low_rank_model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss=CategoricalCrossentropy(),
        metrics=[CategoricalAccuracy()]
    )
    effective_model = create_eff_model(low_rank_model)

    training_ds = training_ds.shuffle(buffer_size=1024).batch(128)

    loss_fn = CategoricalCrossentropy()
    acc_fn = CategoricalAccuracy()
    optimizer = RMSprop()
    val_acc_fn = CategoricalAccuracy(name="val_categorical_accuracy")
    for epoch in range(50):
        acc_fn.reset_state()
        val_acc_fn.reset_state()

        print(f"Starting epoch: {epoch}")

        for step, (x_batch, y_batch) in enumerate(training_ds):
            with tf.GradientTape() as tape:
                y_pred = low_rank_model(x_batch)
                err = loss_fn(y_batch, y_pred)
                loss = tf.reduce_mean(err)
                acc_fn.update_state(y_batch, y_pred)
            grads = tape.gradient(loss, low_rank_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, low_rank_model.trainable_weights))
            print(f"\rStep {step} - loss: {loss:.4f} acc: {acc_fn.result().numpy():.3f}", end="")
        print()

        set_eff_weights(effective_model, low_rank_model)
        indices = list(range(len(x_train)))
        random.shuffle(indices)
        indices = indices[:256]
        x_batch = x_train[indices]
        y_batch = y_train[indices]
        with tf.GradientTape() as tape:
            lr_y_pred = low_rank_model(x_batch)
            lr_err = loss_fn(y_batch, lr_y_pred)
            lr_loss = tf.reduce_mean(lr_err)
            lr_grads = tape.gradient(lr_loss, low_rank_model.trainable_weights)
        eff_grads = eff_gradients(lr_grads, low_rank_model)

        with tf.GradientTape() as tape:
            y_pred = effective_model(x_batch)
            err = loss_fn(y_batch, y_pred)
            loss = tf.reduce_mean(err)
            grads = tape.gradient(loss, effective_model.trainable_weights)
        grads = [g.numpy() for g in grads]

        for i, (eff_g, g) in enumerate(list(zip(eff_grads, grads))[::2]):
            # print(i, np.linalg.norm(eff_g), np.linalg.norm(g))
            dist = np.sum(eff_g * g) ** 2
            # print(i, dist)
            dist /= np.linalg.norm(eff_g) * np.linalg.norm(g)
            # print(i, dist)
            dist = np.arccos(dist)
            print(i, dist / np.pi * 180)

        val_y_pred = low_rank_model(x_test)
        val_err = loss_fn(y_test, val_y_pred)
        val_loss = tf.reduce_mean(val_err)
        val_acc_fn.update_state(y_test, val_y_pred)

        print(f"val loss: {val_loss:.4f} val acc: {val_acc_fn.result().numpy():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a low rank network and compares its "
                                                 "gradients with the effective model's gradients")
    parser.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"])
    args = parser.parse_args()
    main(args.dataset)

import warnings

import tensorflow as tf
from tensorflow.keras.layers import Dense, InputLayer, ReLU, Softmax, Flatten
from tensorflow.keras.models import Sequential, Model

from low_rank_layer import LowRankDense


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
        ReLU(),
        get_dense(),
        ReLU(),
        Dense(output_shape),
        Softmax()
    ])
    model.compile(
        optimizer="rmsprop",
        loss="categorical_crossentropy",
        metrics="accuracy"
    )
    return model


def create_dual_model_skeleton(low_rank_model: Model) -> Model:
    layers = []
    for layer in low_rank_model.layers:
        if isinstance(layer, InputLayer):
            layers.append(InputLayer(input_shape=layer.input_shape))
        elif isinstance(layer, Flatten):
            layers.append(Flatten())
        elif isinstance(layer, ReLU):
            layers.append(ReLU())
        elif isinstance(layer, Softmax):
            layers.append(Softmax())
        elif isinstance(layer, Dense):
            layers.append(Dense(layer.output_shape[-1]))
        elif isinstance(layer, LowRankDense):
            layers.append(Dense(layer.output_shape[-1]))
        else:
            raise ValueError(f"Encountered unrecognized layer of type {type(layer)}")
    dual_model = Sequential(layers)
    dual_model.build(input_shape=low_rank_model.input_shape)
    return dual_model


def set_dual_network_weights(low_rank_network: Model, dual_network: Model):
    if len(low_rank_network.layers) != len(dual_network.layers):
        raise ValueError("Low rank network and dual network should have the same number of layers")

    for low_rank_layer, dual_layer in zip(low_rank_network.layers, dual_network.layers):
        if isinstance(low_rank_layer, LowRankDense) and isinstance(dual_layer, Dense):
            u, v = low_rank_layer.kernel_u, low_rank_layer.kernel_v
            b = low_rank_layer.b
            w = u @ v
            dual_layer.set_weights([w.numpy(), b.numpy()])
        elif isinstance(low_rank_layer, Dense) and isinstance(dual_layer, Dense):
            dual_layer.set_weights([x.numpy() for x in low_rank_layer.weights])
        else:
            if type(low_rank_layer) is not type(dual_layer):
                raise ValueError("Dual network doesn't match with low rank network in structure. "
                                 f"Got {type(low_rank_layer)} and {type(dual_layer)}")


def calc_low_rank_dual_gradients(
    low_rank_network: Model,
    gradients: list[tf.Tensor]
) -> list[tf.Tensor]:
    res = []
    j = 0
    for i, layer in enumerate(low_rank_network.layers):
        if isinstance(layer, Dense):
            res.append(gradients[j].numpy())  # w
            res.append(gradients[j+1].numpy())  # bias
            j += 2
        elif isinstance(layer, LowRankDense):
            u, v = layer.kernel_u.numpy(), layer.kernel_v.numpy()
            g = (u + gradients[j].numpy()) @ (v + gradients[j + 1].numpy())
            g -= u @ v
            res.append(g)  # u, v
            res.append(gradients[j+2].numpy())  # bias
            j += 3
        elif len(layer.weights) != 0:
            raise ValueError(f"Don't know how to calculate low rank dual gradients for layer: "
                             f"{type(layer)}")
    assert j == len(gradients)
    return res

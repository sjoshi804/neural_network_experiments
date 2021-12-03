from typing import List

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Softmax, ReLU

from low_rank_layer import LowRankDense


def create_eff_model(low_rank_model: Sequential, name: str) -> Sequential:
    layers = []
    for layer in low_rank_model.layers:
        if isinstance(layer, InputLayer):
            layers.append(InputLayer(input_shape=layer.input_shape))
        elif isinstance(layer, LowRankDense):
            layers.append(Dense(layer.b.shape[0]))
        elif isinstance(layer, Dense):
            layers.append(Dense(layer.units))
        elif isinstance(layer, Flatten):
            layers.append(Flatten())
        elif isinstance(layer, Softmax):
            layers.append(Softmax())
        elif isinstance(layer, ReLU):
            layers.append(ReLU())
        else:
            raise ValueError(f"Unrecognized layer type: {type(layer)}")
    eff_model = Sequential(layers, name=name)
    eff_model.build(input_shape=low_rank_model.input_shape)
    return eff_model


def set_eff_weights(eff_model: Sequential, low_rank_model: Sequential):
    for eff_layer, low_rank_layer in zip(eff_model.layers, low_rank_model.layers):
        if isinstance(eff_layer, Dense) and isinstance(low_rank_layer, LowRankDense):
            eff_layer.set_weights(low_rank_layer.effective_weights())
        elif isinstance(eff_layer, Dense) and isinstance(low_rank_layer, Dense):
            eff_layer.set_weights(low_rank_layer.get_weights())
        elif isinstance(eff_layer, (InputLayer, Flatten, Softmax, ReLU)) and \
                isinstance(low_rank_layer, (InputLayer, Flatten, Softmax, ReLU)):
            continue
        else:
            raise ValueError(
                f"Unrecognized layers of type: {type(eff_layer)} and {type(low_rank_layer)}")


def eff_gradients(grads: List[np.ndarray], low_rank_model: Sequential) -> List[np.ndarray]:
    eff_grads = []
    grad_ind = 0
    for layer in low_rank_model.layers:
        if isinstance(layer, Dense):
            w_grads, b_grads = grads[grad_ind:grad_ind + 2]
            eff_grads.extend([w_grads.numpy(), b_grads.numpy()])
            grad_ind += 2
        elif isinstance(layer, LowRankDense):
            b_grads = grads[grad_ind].numpy()
            grad_ind += 1
            assert len(layer.kernel_us) == len(layer.kernel_vs)
            rank_blocks = len(layer.kernel_us)
            grad_end = grad_ind + 2 * rank_blocks
            u_grads = np.concatenate(grads[grad_ind:grad_end:2], axis=1)
            v_grads = np.concatenate(grads[grad_ind + 1:grad_end:2], axis=0)
            u = layer.kernel_u().numpy()
            v = layer.kernel_v().numpy()
            eff_grad = (u + u_grads) @ (v + v_grads) - u @ v
            eff_grads.extend([eff_grad, b_grads])
            grad_ind = grad_end
        else:
            # print(f"Skipping layer: {type(layer)}")
            continue
    return eff_grads

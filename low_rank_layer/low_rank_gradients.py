import argparse

import numpy as np
import pandas as pd
import pprint

import tensorflow as tf

from data import load_data
from model import get_model, create_dual_model_skeleton, set_dual_network_weights, calc_low_rank_dual_gradients


def main(dataset: str):
    (x_train, y_train), (x_test, y_test) = load_data(dataset)

    for rank in [100]:
        model = get_model(x_train.shape[1:], y_train.shape[-1], dim=512, rank=rank)
        dual_model = create_dual_model_skeleton(model)

        model.summary()
        dual_model.summary()

        gradient_diffs = []

        for epoch in range(100):
            model.fit(x_train, y_train, batch_size=256, validation_data=(x_test, y_test),
                      steps_per_epoch=1)
            set_dual_network_weights(model, dual_model)
            with tf.GradientTape() as tape:
                pred = model(x_train)
                loss = -tf.reduce_mean(y_train * tf.math.log(pred))
                grads = tape.gradient(loss, model.weights)
                grads = calc_low_rank_dual_gradients(model, gradients=grads)
            with tf.GradientTape() as tape:
                pred = dual_model(x_train)
                loss = -tf.reduce_mean(y_train * tf.math.log(pred))
                dual_grads = tape.gradient(loss, dual_model.weights)

            diffs = {"epoch": epoch, "loss": loss.numpy()}
            for i, (g, dg) in enumerate(zip(grads, dual_grads)):
                diff = g / np.linalg.norm(g) - dg / np.linalg.norm(dg)
                norm = np.linalg.norm(diff)
                if np.isnan(norm):
                    import pdb; pdb.set_trace()
                diffs[f"layer {i} diff"] = norm
            pprint.pprint(diffs)
            gradient_diffs.append(diffs)

        gradient_diffs = pd.DataFrame(gradient_diffs).set_index("epoch")
        gradient_diffs.to_csv(f"gradients_rank{rank}_{dataset}.csv", index="epoch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runs a low rank dense layer experiment")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["mnist", "cifar10", "cifar100"]
    )
    args = parser.parse_args()
    main(dataset=args.dataset)

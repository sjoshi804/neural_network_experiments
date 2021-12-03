import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class LowRankDense(layers.Layer):
    def __init__(self, num_outputs, init_rank=1):
        super(LowRankDense, self).__init__()
        self.num_outputs = num_outputs
        self.num_inputs = None

        self.init_rank = init_rank
        self.rank = 0
        self.kernel_us = []
        self.kernel_vs = []
        self.b = None

    def build(self, input_shape):
        self.num_inputs = int(input_shape[-1])
        b_init = tf.zeros_initializer()
        self.b = self.add_weight(
            shape=(self.num_outputs,),
            initializer=b_init,
            trainable=True,
            name="bias"
        )
        self.add_rank(self.init_rank)  # initial ranks

        def orth_reg():
            tot_reg = 0
            for i, kernel_u in enumerate(self.kernel_us):
                for prev_kernel_u in self.kernel_us[:i]:
                    reg = tf.matmul(prev_kernel_u, kernel_u, transpose_a=True)
                    reg -= tf.eye(reg.shape[0], reg.shape[1])
                    tot_reg += tf.nn.l2_loss(reg)
            for i, kernel_v in enumerate(self.kernel_vs):
                for prev_kernel_v in self.kernel_vs[:i]:
                    reg = tf.matmul(prev_kernel_v, kernel_v, transpose_b=True)
                    reg -= tf.eye(reg.shape[0], reg.shape[1])
                    tot_reg += tf.nn.l2_loss(reg)
            return tot_reg

        self.add_loss(orth_reg)

    def add_rank(self, num_ranks):
        w_init = tf.random_normal_initializer()
        kernel_u = self.add_weight(
            shape=(self.num_inputs, num_ranks),
            initializer=w_init,
            trainable=True,
            name=f"u_rank-{self.rank}-{self.rank + num_ranks}"
        )
        kernel_v = self.add_weight(
            shape=(num_ranks, self.num_outputs),
            initializer=w_init,
            trainable=True,
            name=f"v_rank-{self.rank}-{self.rank + num_ranks}"
        )
        self.kernel_us.append(kernel_u)
        self.kernel_vs.append(kernel_v)
        self.rank += num_ranks

    def kernel_u(self):
        return tf.concat(self.kernel_us, axis=1)

    def kernel_v(self):
        return tf.concat(self.kernel_vs, axis=0)

    def set_rank(self, ranks: int):
        w = self.kernel_u() @ self.kernel_v()
        u, s, v = np.linalg.svd(w, full_matrices=False)
        s = s[:ranks]
        u = u[:, :ranks] * s
        v = v[:ranks, :] * s[:, None]
        self.kernel_us = [
            self.add_weight(
                shape=(self.num_inputs, ranks),
                initializer=lambda *args, **kwargs: u,
                trainable=True,
                name=f"u_rank-{self.rank}-{self.rank + ranks}"
            )
        ]
        self.kernel_vs = [
            self.add_weight(
                shape=(ranks, self.num_outputs),
                initializer=lambda *args, **kwargs: v,
                trainable=True,
                name=f"v_rank-{self.rank}-{self.rank + ranks}"
            )
        ]

    def call(self, inputs, *args, **kwargs):
        return inputs @ (self.kernel_u() @ self.kernel_v()) + self.b

    def effective_weights(self):
        return [(self.kernel_u() @ self.kernel_v()).numpy(), self.b.numpy()]

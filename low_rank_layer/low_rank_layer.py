import tensorflow as tf
from tensorflow.keras.layers import Layer


class LowRankDense(Layer):
    def __init__(self, num_outputs, rank=1):
        super(LowRankDense, self).__init__()
        self.num_outputs = num_outputs
        self.rank = rank
        self.kernel_u = None
        self.kernel_v = None
        self.b = None

    def build(self, input_shape):
        num_inputs = int(input_shape[-1])
        w_init = tf.random_normal_initializer()
        self.kernel_u = tf.Variable(
            initial_value=w_init(shape=(num_inputs, self.rank), dtype="float32"),
            trainable=True
        )
        self.kernel_v = tf.Variable(
            initial_value=w_init(shape=(self.rank, self.num_outputs), dtype="float32"),
            trainable=True
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.num_outputs,), dtype="float32"),
            trainable=True
        )

        eye = tf.eye(self.rank)
        u_reg = tf.matmul(self.kernel_u, self.kernel_u, transpose_a=True) - eye
        v_reg = tf.matmul(self.kernel_v, self.kernel_v, transpose_b=True) - eye
        loss = tf.norm(u_reg) + tf.norm(v_reg)
        self.add_loss(0.001 * loss)

    def call(self, inputs, *args, **kwargs):
        return inputs @ (self.kernel_u @ self.kernel_v) + self.b


class Dense(Layer):
    def __init__(self, num_outputs):
        super(Dense, self).__init__()
        self.num_outputs = num_outputs
        self.kernel = None
        self.b = None

    def build(self, input_shape):
        num_inputs = int(input_shape[-1])
        w_init = tf.random_normal_initializer()
        self.kernel = tf.Variable(
            initial_value=w_init(shape=(num_inputs, self.num_outputs), dtype="float32"),
            trainable=True
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.num_outputs,), dtype="float32"),
            trainable=True
        )

        u_reg = tf.matmul(self.kernel, self.kernel, transpose_a=True) - tf.eye(self.num_outputs)
        v_reg = tf.matmul(self.kernel, self.kernel, transpose_b=True) - tf.eye(num_inputs)
        loss = tf.norm(u_reg) + tf.norm(v_reg)
        self.add_loss(0.001 * loss)

    def call(self, inputs, *args, **kwargs):
        return inputs @ self.kernel + self.b

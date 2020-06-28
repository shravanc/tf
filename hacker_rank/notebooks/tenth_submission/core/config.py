import tensorflow as tf


class Configuration:
    def __init__(self):
        self.shuffle_buffer = 1000
        self.pca_components = 16
        self.batch_size = 32
        self.epochs = 1 #1000

        self.activation = tf.keras.activations.sigmoid
        self.loss = tf.keras.losses.Huber(delta=1.0)
        self.lr_rate = 0.005
        self.optimizer = tf.keras.optimizers.SGD(lr=self.lr_rate, momentum=0.9)
        self.metrics = ['mse']

import tensorflow as tf
from tensorflow.keras import regularizers


def build_model(conf):
    model = tf.keras.Sequential([

        tf.keras.layers.Dense(576, activation=conf.activation, kernel_regularizer=regularizers.l2(0.005)),
        tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(416, activation=conf.activation, kernel_regularizer=regularizers.l2(0.0009)),
        # tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(992, activation=conf.activation, kernel_regularizer=regularizers.l2(0.001)),
        # tf.keras.layers.Dropout(0.1),

        tf.keras.layers.Dense(992, activation=conf.activation, kernel_regularizer=regularizers.l2(0.0007)),

        tf.keras.layers.Dense(1)
    ])

    # model = tf.keras.Sequential([
    #
    #     tf.keras.layers.Dense(864, activation=conf.activation, kernel_regularizer=regularizers.l2(0.007)),
    #     tf.keras.layers.Dropout(0.5),
    #
    #     tf.keras.layers.Dense(352, activation=conf.activation, kernel_regularizer=regularizers.l2(0.004)),
    #     tf.keras.layers.Dropout(0.4),
    #
    #     tf.keras.layers.Dense(32, activation=conf.activation, kernel_regularizer=regularizers.l2(0.0001)),
    #     tf.keras.layers.Dropout(0.5),
    #
    #     tf.keras.layers.Dense(32, activation=conf.activation, kernel_regularizer=regularizers.l2(0.0001)),
    #     tf.keras.layers.Dropout(0.5),
    #
    #     tf.keras.layers.Dense(992, activation=conf.activation, kernel_regularizer=regularizers.l2(0.0007)),
    #     tf.keras.layers.Dropout(0.5),
    #
    #     tf.keras.layers.Dense(1)
    # ])


    return model

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow_docs as tfdocs
import os
import calendar
import time
ts = calendar.timegm(time.gmtime())


from tensorflow.keras import layers
from tensorflow.keras import regularizers

train_path = '../../clean_train.csv'
df = pd.read_csv(train_path)

train_df = df.sample(frac=0.85, random_state=2)
valid_df = df.drop(train_df.index)

TARGET = 'Attrition_rate'
train_labels = train_df.pop(TARGET)
valid_labels = valid_df.pop(TARGET)

# Scale
stats = train_df.describe().transpose()
train_df = (train_df - stats['mean']) / stats['std']
valid_df = (valid_df - stats['mean']) / stats['std']

# Convert to numpy
train_data = train_df.to_numpy()
valid_data = valid_df.to_numpy()


# Converting to tf.data
def prepare_dataset(data, labels, batch, shuffle_buffer):
    # data = tf.expand_dims(data, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)  # .repeat()
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


batch_size = 32
buffer = 1000
epochs = 100
lr_rate = 1e-2
momentum = 0.9
STEPS_PER_EPOCH = len(train_df) // batch_size

train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)



def get_optimizer():
    return tf.keras.optimizers.SGD()


def compile_and_fit(model):
    optimizer = get_optimizer()
    model.compile(
        loss=tf.keras.losses.mse,
        optimizer=optimizer,
        metrics=['mse', 'mae']
    )

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=valid_dataset
    )

    return history


# Tiny Model
tiny = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

compile_and_fit(tiny)
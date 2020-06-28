import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


train_path = './clean_train.csv'
train_df = pd.read_csv(train_path)

TARGET = 'Attrition_rate'
train_labels = train_df.pop(TARGET)

# Scale
stats = train_df.describe().transpose()
train_df = (train_df-stats['mean'])/stats['std']


train_data = train_df.to_numpy()


# Converting to tf.data
def prepare_dataset(data, labels, batch, shuffle_buffer):
    #data = tf.expand_dims(data, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


batch_size = 32
buffer = 1000
epochs = 300
lr_rate = 1e-2
momentum = 0.9

train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)

depth_level = "lr_3_1"
# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1028, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.mse,
    optimizer=tf.keras.optimizers.SGD(lr=lr_rate, momentum=momentum),
    metrics=['mse', 'mae']
)


history = model.fit(
    train_dataset,
    epochs=epochs,
)

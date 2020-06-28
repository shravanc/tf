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


# ===========================================================================


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
# tiny = tf.keras.Sequential([
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])
#
# collection = {'tiny': compile_and_fit(tiny)}
collection = {}
# # Small Model
# small = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])
#
# collection['small'] = compile_and_fit(small)
#
# # Medium Model
# medium = tf.keras.Sequential([
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])
#
# collection['medium'] = compile_and_fit(medium)
#
# # Large Model
# large = tf.keras.Sequential([
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])
#
# collection['large'] = compile_and_fit(large)
# Extreme Model
# extreme = tf.keras.Sequential([
#     tf.keras.layers.Dense(1028, activation='relu'),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])
#
# collection['extreme'] = compile_and_fit(extreme)

# Extreme Regularised
# one_reg_extreme = tf.keras.Sequential([
#     tf.keras.layers.Dense(1028, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])
# collection['one_drop'] = compile_and_fit(one_reg_extreme)

# two_reg_extreme = tf.keras.Sequential([
#     tf.keras.layers.Dense(1028, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])
# collection['two_drop'] = compile_and_fit(two_reg_extreme)
#
# three_reg_extreme = tf.keras.Sequential([
#     tf.keras.layers.Dense(1028, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dense(1)
# ])
# collection['three_drop'] = compile_and_fit(three_reg_extreme)
#

full_reg_extreme = tf.keras.Sequential([
    tf.keras.layers.Dense(1028, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])
collection['full_drop'] = compile_and_fit(full_reg_extreme)


double_reg_extreme = tf.keras.Sequential([
    tf.keras.layers.Dense(1028, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])
collection['double'] = compile_and_fit(double_reg_extreme)

double_reg_extreme_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1028, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1)
])
collection['double_2'] = compile_and_fit(double_reg_extreme_2)

double_reg_extreme_l1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1028, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.001)),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1)
])
collection['double_l1'] = compile_and_fit(double_reg_extreme_l1)


double_reg_extreme_l1_l2 = tf.keras.Sequential([
    tf.keras.layers.Dense(1028, activation='relu', kernel_regularizer=regularizers.l1_l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l1_l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l1_l2(0.001)),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(1)
])
collection['double_l1_l2'] = compile_and_fit(double_reg_extreme_l1_l2)


plt.figure(figsize=(15, 10))

image_path = '/home/shravan/Desktop/model_plots/'
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
for i, key in enumerate(list(collection.keys())):
    metric = collection[key].history['mse']
    val_metric = collection[key].history['val_mse']

    # metric = metric[-50:]
    # val_metric = metric[-50:]

    epochs = range(len(metric))

    plt.plot(epochs, metric, colors[i], label=f"{key}_Training MSE")
    plt.plot(epochs, val_metric, colors[i], linestyle='dashed', label=f"{key}_Validation MSE")

plt.legend()
x = ts
image = os.path.join(image_path, f"{x}_image")
plt.show()



forecast = []


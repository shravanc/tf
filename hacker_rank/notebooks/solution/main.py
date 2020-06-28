import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers

train_path = '../../clean_train.csv'
df = pd.read_csv(train_path)
df.pop('Unnamed: 0')
train_df = df[:6500] #df.sample(frac=0.6, random_state=2)
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
epochs = 1000
lr_rate = 1e-2
momentum = 0.9
STEPS_PER_EPOCH = len(train_df) // batch_size

train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)

# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1_l2(0.001)),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1_l2(0.001)),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(1)
# ])


# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)
])

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
  0.001,
  decay_steps=STEPS_PER_EPOCH*1000,
  decay_rate=1,
  staircase=False)

model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.SGD(lr_schedule),
    metrics=['mae']
)

history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset
)

plots = ['mae', 'loss']
for plot in plots:
    metric = history.history[plot]
    val_metric = history.history[f"val_{plot}"]
    epochs = range(len(metric))

    plt.figure(figsize=(15, 10))
    plt.plot(epochs, metric, label=f"Training {plot}")
    plt.plot(epochs, val_metric, label=f"Validation {plot}")
    plt.legend()
    plt.title(f"Training and Validation for {plot}")

    file_name = f"tr_va_{plot}"
    plt.show()

# Forecasting
forecast = []
for index, row in valid_df.iterrows():
    data = row.values[np.newaxis]
    result = model.predict(data)
    forecast.append(result[0][0])

original_data = valid_labels[-50:]
forecast_data = forecast[-50:]
x_axis = range(len(original_data))

plt.figure(figsize=(15, 10))
plt.plot(x_axis, original_data, label="Original Data")
plt.plot(x_axis, forecast_data, label="Prediction Data")
plt.legend()
plt.title("Forecast Data")
plt.show()

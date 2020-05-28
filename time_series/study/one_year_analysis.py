import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

url = 'https://raw.githubusercontent.com/shravanc/datasets/master/one_year_timeseries.csv'
df = pd.read_csv(url)

mean = df['series'].mean()
std = df['series'].std()

df['n_series'] = (df['series'] - mean) / std

print(df.head(2))

series = df['n_series'].values
time = range(len(series))

split_time = 8000

train_data = series[:split_time]
valid_data = series[split_time:]

time_train = time[:split_time]
time_valid = time[split_time:]

plt.figure(figsize=(20, 10))
plt.plot(time_train, train_data, label=['Train Data'])
plt.plot(time_valid, valid_data, label=['Valid Data'])
plt.title('One Year Data')
plt.show()

ws = 24
bs = 32
sb = 1000


def windowed_dataset(t_series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(t_series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


train_set = windowed_dataset(train_data, ws, bs, sb)
valid_set = windowed_dataset(valid_data, ws, bs, sb)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=[ws]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1),
    tf.keras.layers.Lambda(lambda x: x*100.)
])

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['accuracy', 'mae']
)

history = model.fit(
    train_set,
    epochs=10,
    validation_data=valid_set
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(20, 10))
plt.plot(epochs, acc, label=['Training Accuracy'])
plt.plot(epochs, val_acc, label=['Validation Accuracy'])
plt.title('Training and Validation Accuracy')
plt.show()

plt.figure(figsize=(20, 10))
plt.plot(epochs, loss, label=['Training Loss'])
plt.plot(epochs, val_loss, label=['Validation Loss'])
plt.title('Training and Validation Loss')
plt.show()

forecast = []
print("***Forecasting***")
for time in range(len(series) - ws):
    pred_input = series[time:time + ws][np.newaxis]
    prediction = model.predict(pred_input)
    forecast.append(prediction)

print("***Forecasting***")
forecast = forecast[split_time - ws:]
results = np.array(forecast)[:, 0, 0]

print(len(valid_data))
print(len(results))
print(len(time_valid))

print(results[0:20])
print(valid_data[0:20])

plt.figure(figsize=(20, 10))
#plt.plot(time_valid, valid_data, 'Original Data')
plt.plot(time_valid, results, 'Prediction Data')
plt.title('Prediction Plot')
plt.show()

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/shravanc/datasets/master/nov_time_series.csv'
df = pd.read_csv(url)
mean = df['series'].mean()
std = df['series'].std()

df['n_series'] = (df['series'] - mean) / std

series = df['n_series'].values
time = range(len(series))

split_time = 500

train_data = series[:split_time]
valid_data = series[split_time:]

time_train = time[:split_time]
time_valid = time[split_time:]

ws = 24
bs = 2
sb = 500


def windowed_dataset(t_series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(t_series)
    dataset = dataset.window(window_size + 24, shift=24, drop_remainder=True)
    dataset = dataset.flat_map(lambda x: x.batch(window_size + 24))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-24], window[-24:]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


train_set = windowed_dataset(
    train_data,
    ws,
    bs,
    sb
)

valid_set = windowed_dataset(
    valid_data,
    ws,
    bs,
    sb
)

for x, y in train_set.batch(1):
    print("x: ", x.numpy())
    print("Y: ", y.numpy())


model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=[ws]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(24)
])

model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['accuracy']
)

history = model.fit(
    train_set,
    epochs=100,
    validation_data=valid_set
)

print(len(series))
# print(series[0:23])

forecast = []
for time in range(len(series - ws)):
    pred_data = series[time:time + ws][np.newaxis]
    if len(pred_data[0]) < 24:
        continue
    prediction = model.predict(pred_data)
    forecast.append(prediction)
    # print(prediction)

forecast = forecast[split_time - ws:]
results = np.array(forecast)[:, 0, 0]

print(len(forecast))
print(len(results))
print(len(valid_data))

plt.figure(figsize=(20, 10))
plt.plot(time_valid, valid_data, label=['Validation Data'])
plt.plot(time_valid, results[:-1], label=['Prediction Data'])
plt.show()


d = "3557.033333,2848.9833329999997,1294.0805560000001,19.42222222,0.0,0.0,0.0,0.0,0.0,3816.338889,0.0,0.0,3789.961111,1224.0777779999999,17.77777778,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3189.275".split(',')
pr_data = [float(s) for s in d]
print(pr_data)
print(model.predict([pr_data]))
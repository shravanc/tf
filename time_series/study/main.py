import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

url = 'https://raw.githubusercontent.com/shravanc/datasets/master/nov_time_series.csv'
# url = 'https://raw.githubusercontent.com/shravanc/datasets/master/one_year_timeseries.csv'
df = pd.read_csv(url)

mean = df['series'].mean()
std = df['series'].std()

df['normalised_series'] = (df['series']-mean)/std
print(df.head(3))

series = df['normalised_series'].values
time = range(len(series))

print(len(series))
print(len(time))

total = 720
split_time = 500

train_data = series[:split_time]
valid_data = series[split_time:]

time_train = time[:split_time]
time_valid = time[split_time:]

plt.figure(figsize=(20, 10))
plt.plot(time_train, train_data, label=['Train Data'])
plt.plot(time_valid, valid_data, label=['Validation Data'])
plt.title('November Data')
# plt.show()


ws = 24
bs = 20
sb = 500


def windowed_dataset(t_series, window_size, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices(t_series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
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

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=[ws]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])


model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['accuracy', 'mae']
)

history = model.fit(
    train_set,
    epochs=500,
    validation_data=valid_set
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(len(acc))

plt.figure(figsize=(20, 10))
plt.plot(epochs, acc, label=['Training Accuracy'])
plt.plot(epochs, val_acc, label=['Validation Accuracy'])
plt.title('November Data')
plt.show()


forecast = []
print("*****FORECASTING*****")

"""
720
720-24=696

pred_data = 0..23, 2..24, 3..25
"""
for time in range(len(series-ws)):
    pred_data = series[time:time+ws][np.newaxis]
    if len(pred_data[0]) < 24:
        continue
    prediction = model.predict(pred_data)
    forecast.append(prediction)


forecast = forecast[split_time-ws:]
#r = np.array(forecast)


results = np.array(forecast)[:, 0, 0]
#results.pop()
print(len(results))
print(len(time_valid))


plt.figure(figsize=(20,10))
plt.plot(time_valid, valid_data, label=['Original Data'])
plt.plot(time_valid, results[:-1], label=['Prediction Data'])
plt.title('Prediction Plot')
plt.show()


d = "3557.033333,2848.9833329999997,1294.0805560000001,19.42222222,0.0,0.0,0.0,0.0,0.0,3816.338889,0.0,0.0,3789.961111,1224.0777779999999,17.77777778,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3189.275".split(',')
pr_data = [float(s) for s in d]
print(pr_data)
print(model.predict([pr_data]))

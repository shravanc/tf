import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

series = list(range(0, 1001))
time = series




def plot_series(series, time):
  plt.plot(time, series, "-")
  plt.xlabel("Time")
  plt.ylabel("Series")
  plt.grid(True)



plot_series(series, time)
#plt.show()


ws = 4
bs = 32
split_time = 600

x_train = series[:split_time]
time_tr = time[:split_time]

x_valid = series[split_time:]
time_va = time[split_time:]


def get_dataset(series, ws, bs=16):
  #dataset = tf.data.Dataset.range(5001)
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(ws+1)
  dataset = dataset.flat_map(lambda window: window.batch(ws+1))
  dataset = dataset.map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.shuffle(500)
  dataset = dataset.batch(bs)

  return dataset



dataset = get_dataset(x_train, ws, bs)
#for i, (x, y) in enumerate(dataset):
#  print(f"x: {x}, y: {y}")



## model START
l0 = tf.keras.layers.Dense(1)
model = tf.keras.Sequential([l0])


model.compile(
  loss="mse",
  optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9),
  metrics=['mse', 'mae']
)


history = model.fit(
  dataset,
  epochs=100
)

## model END

forecast = []
# iterating from 0 to (1000-ws(4))
# iterates from 0 to 996
for time in range(len(series) - ws):
  input_to_predict = [series[time:time + ws]] #[np.newaxis]
  prediction = model.predict( input_to_predict )
  forecast.append(prediction)

#print(forecast)
#print(np.array(forecast))
#print(np.array(forecast)[:, 0, 0])



forecast = forecast[split_time-ws:]
results = np.array(forecast)[:, 0, 0]

print("*****")
print(len(results))
print(results)
print("*****")

plt.figure(figsize=(10, 6))
plot_series(time_va, x_valid)
plot_series(time_va, results)
plt.show()

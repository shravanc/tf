import tensorflow as tf
import pandas as pd
from lib.utils import plot_series
import numpy as np
import matplotlib.pyplot as plt



path = "https://raw.githubusercontent.com/shravanc/datasets/master/nov_time_series.csv"
path = "https://raw.githubusercontent.com/shravanc/datasets/master/one_year_timeseries.csv"
df = pd.read_csv(path)
series = df['series'].values

def split_data(df):
  tr = df.sample(frac=0.8, random_state=0)
  te = df.drop(tr.index)

  return tr, te

def normalize(df):
  copy = df.copy()
  series = copy['series'].values
  mean = copy['series'].mean()
  std =  copy['series'].std()
  copy['series'] = (series-mean) / std

  return copy

def windowed_dataset(dataset, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(dataset)
  dataset = dataset.window(window_size+1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size+1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)

  return dataset

tr_data, te_data = split_data(df)
print(tr_data.tail())
norm_tr_data = normalize(tr_data)
norm_te_data = normalize(te_data)

window_size = 24
batch_size  = 32
shuffle_buffer = 500

tr_dataset = windowed_dataset(norm_tr_data['series'].values, window_size, batch_size, shuffle_buffer)
te_dataset = windowed_dataset(norm_te_data['series'].values, window_size, batch_size, shuffle_buffer)


def build_model():
  l0 = tf.keras.layers.Dense(1, input_shape=[window_size])
  model = tf.keras.models.Sequential([l0])
  return l0, model

l0, model = build_model()
model.summary()

model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9), metrics=['mae', 'mse'])
history = model.fit(tr_dataset, epochs=100)
print(history)


forecast = []
for time in range(len(series) - window_size):
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))

#forecast = forecast[split_time-window_size:]
results = np.array(forecast)[:, 0, 0]

plt.figure(figsize=(10, 6))

plot_series(time, x_valid)
plot_series(time_valid, results)
plt.show()




d = "3557.033333,2848.9833329999997,1294.0805560000001,19.42222222,0.0,0.0,0.0,0.0,0.0,3816.338889,0.0,0.0,3789.961111,1224.0777779999999,17.77777778,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3189.275".split(',')
pr_data = [float(s) for s in d]
print(pr_data)
prediction = model.predict([pr_data])
print(prediction)
#tf.keras.metrics.mean_absolute_error(norm_te_data['series'].values, results).numpy()

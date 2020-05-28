import pandas as pd
import tensorflow as tf
from lib.utils import plot_series
import numpy as np
import matplotlib.pyplot as plt


path = "https://raw.githubusercontent.com/shravanc/datasets/master/nov_time_series.csv"

df = pd.read_csv(path) #, names=["series"])
series = df['series'].values
time   = list(range(0, len(series)))

values = df['series'].values
dataset = tf.data.Dataset.from_tensor_slices(values)
dataset = dataset.window(25, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(25))
dataset = dataset.map(lambda window: (window[:-1], window[-1: ]))
for x, y in dataset:
  print(x.numpy(), y.numpy())
  break


plt.figure(figsize=(15, 10))
plot_series(time, series)
plt.show()


import pandas as pd
import os
import matplotlib.pyplot as plt
import glob
import tensorflow as tf
import numpy as np


url = 'https://raw.githubusercontent.com/shravanc/datasets/master/timeseries.csv'
df = pd.read_csv(url)
grouped = df.groupby('PeriodStart')
groups  = list(grouped)

x1 = []
y1 = []

for i, group in enumerate(groups):

  try:
    df1 = group[1]
    df1 = df1.sort_values(by='hours')

    df2 = groups[i+1][1]
    df2 = df2.sort_values(by='hours')

    y1.append( list(df2['PV output'].values) )
    x1.append( list(df1['PV output'].values) )
  except:
    print(i)
  

import itertools
x = list(itertools.chain.from_iterable(x1))
y = list(itertools.chain.from_iterable(y1))

x = x[0:240]
y = y[0:240]

print(x[24:48])
print(y[0:24])

"""
dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.window(48, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(48))
dataset = dataset.map(lambda window: (window[25:48], window[25:48]))
for x,y in dataset:
  print(x.numpy(), y.numpy())
  break
"""



dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.window(48, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(48))
dataset = dataset.map(lambda window: (window[0:24], window[24:48]))
#dataset = dataset.shuffle(buffer_size=10)
#dataset = dataset.batch(2).prefetch(1)
"""
for x,y in dataset:
  print("**x = ", x.numpy())
  print("**y = ", y.numpy())
  print("--->", len(x.numpy()), len(y.numpy()))
  print("--->", len(x.numpy()[0]), len(y.numpy()[0]))
  print("--->", len(x.numpy()[1]), len(y.numpy()[1]))
"""


#l0 = tf.keras.layers.Dense(1, input_shape=[24])
l0 = tf.keras.layers.Dense(24, input_shape=[1])
model = tf.keras.models.Sequential([l0])


model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model.fit(dataset,epochs=100,verbose=0)

print("Layer weights {}".format(l0.get_weights()))



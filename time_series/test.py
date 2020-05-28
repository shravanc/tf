import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""
dataset = tf.data.Dataset.range(50)
dataset = dataset.window(48, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(48))
dataset = dataset.map(lambda window: (window[0:24], window[24:48]))
for x,y in dataset:
  print(x.numpy(), y.numpy())
"""

#"""
dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x,y in dataset:
  print(x.numpy(), y.numpy())
#"""

#l0 = tf.keras.layers.Dense(24, input_shape=[1])
l0 = tf.keras.layers.Dense(1) #, input_shape=[1])
model = tf.keras.models.Sequential([l0])


model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9))
model.fit(dataset,epochs=100,verbose=0)

print("Layer weights {}".format(l0.get_weights()))
dataset = tf.data.Dataset.range(5)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for x,y in dataset:
  print(x.numpy(), y.numpy())
p = model.predict(dataset)
print(p)



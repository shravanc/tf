from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

path_to_file = "./winequality-red.csv"
df = pd.read_csv(path_to_file, sep=",")
print(df.head(20))


train, test = train_test_split(df, test_size=0.2)
train, val   = train_test_split(train, test_size=0.2)

def df_to_dataset(df, shuffle=True, batch_size=32):
  df      = df.copy()
  labels  = df.pop('quality')
  ds      = tf.data.Dataset.from_tensor_slices((dict(df), labels))

  if shuffle:
    ds = ds.shuffle(buffer_size=len(df))

  ds = ds.batch(batch_size)
  return ds


batch_size  = 5
train_ds    = df_to_dataset(train, shuffle=True, batch_size=batch_size)
val_ds      = df_to_dataset(val, shuffle=True, batch_size=batch_size)
test_ds     = df_to_dataset(test, shuffle=True, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
  print("dnesity feature", feature_batch['density'])

example_batch = next(iter(train_ds))[0]

def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())

dioxide = feature_column.numeric_column('free_sulfur_dioxide')
demo(dioxide)


dioxide_buckets = feature_column.bucketized_column(dioxide, boundaries=[2, 6, 10, 14, 18, 24, 30, 50])

demo(dioxide_buckets)

#    fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  alcohol  quality
#0             7.4             0.700         0.00             1.9      0.076                 11.0                  34.0   0.9978  3.51       0.56      9.4        5

feature_columns = []
for header in ['fixed_acidity', 'volatile_acidity', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']:
  d = feature_column.numeric_column(header)
  print("****Demo Begins****", header)
  demo(d)
  feature_columns.append(d)

#feature_columns.append(dioxide_buckets)


feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

print(test.head())

batch_size = 32
train_ds   = df_to_dataset(train, batch_size=batch_size)
val_ds     = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds    = df_to_dataset(test, shuffle=False, batch_size=batch_size)
print(test_ds.take(1))

model = tf.keras.Sequential([
  feature_layer,
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              )

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)


loss, accuracy = model.evaluate(test_ds)
#print("Accuracy--->", accuracy)

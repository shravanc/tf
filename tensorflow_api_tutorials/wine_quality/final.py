from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


path_to_file = "./winequality-red.csv"
df = pd.read_csv(path_to_file)

train, test = train_test_split(df, test_size=0.2)
train, val  = train_test_split(train, test_size=0.2)

def df_to_dataset(df, shuffle=True, batch_size=32):
  df = df.copy()
  labels = df.pop('quality')
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

  if shuffle:
    ds.shuffle(buffer_size=len(df))

  ds = ds.batch(batch_size)
  return ds


batch_size=32
train_ds = df_to_dataset(train, shuffle=True , batch_size=batch_size)
val_ds   = df_to_dataset(val  , shuffle=False, batch_size=batch_size)
test_ds  = df_to_dataset(test , shuffle=False, batch_size=batch_size)


feature_columns = []
for header in ['fixed_acidity', 'volatile_acidity', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']:
  feature_columns.append(feature_column.numeric_column(header))


feature_layers = tf.keras.layers.DenseFeatures(feature_columns)

model = tf.keras.Sequential([
        feature_layers,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam",
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)

print(accuracy)

# https://www.tensorflow.org/tutorials/structured_data/feature_columns

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


#URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
#dataframe = pd.read_csv(URL)
path_to_file = "/home/shravan/tf/tf/tensorflow_api_tutorials/wine_quality/winequality-red.csv"
dataframe = pd.read_csv(path_to_file, sep=",")

print(dataframe.head())

train, test = train_test_split(dataframe, test_size=0.2)
train, val   = train_test_split(train, test_size=0.2)

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('quality')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  print(list(ds.as_numpy_iterator())[0])
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds



# Choosing which column to use:
feature_columns = []

#numeric_columns
for header in ['fixed_acidity', 'volatile_acidity', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']:
  feature_columns.append(feature_column.numeric_column(header))




# feature layer
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

batch_size = 32
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds   = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds  = df_to_dataset(test, shuffle=False, batch_size=batch_size)



#Create, Compile and train the model

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(128, activation='relu'),
  layers.Dense(1)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=5)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy", accuracy)

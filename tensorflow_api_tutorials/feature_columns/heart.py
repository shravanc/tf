from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


URL = 'https://storage.googleapis.com/applied-dl/heart.csv'
dataframe = pd.read_csv(URL)

#path_to_file = "/home/shravan/tf/tf/tensorflow_api_tutorials/wine_quality/winequality-red.csv"
#dataframe = pd.read_csv(path_to_file, sep=";")
print(dataframe.head())


train, test = train_test_split(dataframe, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# Creating Input pipeline using tf.data

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  #labels = dataframe.pop('quality')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  print(list(ds.as_numpy_iterator())[0])
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size  = 5 # A small batch sized is used for demonstration purposes
train_ds    = df_to_dataset(train, shuffle=True,  batch_size=batch_size)
val_ds      = df_to_dataset(val,   shuffle=False, batch_size=batch_size)
test_ds     = df_to_dataset(test,  shuffle=False, batch_size=batch_size)


# Understand the Input Pipeline
for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['age'])
  #print('A batch of ages:', feature_batch['density'])
  print('A batch of targets:', label_batch)


# Demonstrate several type of feature column

example_batch = next(iter(train_ds))[0]

def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())


age = feature_column.numeric_column('age')
#age = feature_column.numeric_column('density')
demo(age)

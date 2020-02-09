from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


path_to_file = "./bank-1.csv"
df = pd.read_csv(path_to_file)
print(df.head())


train, test = train_test_split(df, test_size=0.2)
train, val  = train_test_split(train, test_size=0.2)

def df_to_dataset(df, shuffle=True, batch_size=32):
  df = df.copy()
  labels = df.pop('subscribed')
  print(labels)
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))

  if shuffle:
    ds.shuffle(buffer_size=len(df))
  
  ds = ds.batch(batch_size)
  return ds

batch_size=5
train_ds  = df_to_dataset(train, batch_size=batch_size)
val_ds    = df_to_dataset(val,  shuffle=False, batch_size=batch_size)
test_ds   = df_to_dataset(test, shuffle=False, batch_size=batch_size)

#for feature_batch, label_batch in train_ds.take(1):
#  print(feature_batch.keys())

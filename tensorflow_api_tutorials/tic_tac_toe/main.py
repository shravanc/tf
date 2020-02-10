from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import feature_column
from sklearn.model_selection import train_test_split

URL = "./tic-tac-toe.data.csv"
df = pd.read_csv(URL)
print(df.head())

train, test = train_test_split(df   , test_size=0.2)
train, val  = train_test_split(train, test_size=0.2)


#====================Without Estimator=====================================================
def df_to_dataset(df, shuffle=True, batch_size=32):
  df = df.copy()
  labels = df.pop('class')
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds.shuffle(buffer_size=len(df))

  ds = ds.batch(batch_size)
  return ds

batch_size=32
train_ds = df_to_dataset(train, shuffle=True , batch_size=batch_size)
val_ds   = df_to_dataset(val  , shuffle=False, batch_size=batch_size)
test_ds  = df_to_dataset(test , shuffle=False, batch_size=batch_size)

for train_batch, label_batch in train_ds.take(1):
  print(list(train_batch.keys()))


#=======Feature Column=========
feature_columns = []
for header in ['t_l_s', 't_m_s', 't_r_s', 'm_l_s', 'm_m_s', 'm_r_s', 'b_l_s', 'b_m_s', 'b_r_S']:
  fc = feature_column.categorical_column_with_vocabulary_list(header, df[header].unique())
  feature_columns.append(feature_column.indicator_column(fc))


#=======Feature Column=========

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


model = tf.keras.Sequential([
        feature_layer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=10)

loss, accuracy = model.evaluate(test_ds)
print("Accuracy--->", accuracy)
#====================Without Estimator=====================================================


#====================With Estimator========================================================
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
    if shuffle:
      ds = ds.shuffle(1000)
    ds = ds.batch(batch_size).repeat(num_epochs)
    return ds
  return input_function


y_train = train.pop('class')
train_input_fn = make_input_fn(train, y_train)

y_eval  = val.pop('class')
eval_input_fn = make_input_fn(val, y_eval, num_epochs=1, shuffle=False)


linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

print("Result--->", result)


#====================With Estimator========================================================

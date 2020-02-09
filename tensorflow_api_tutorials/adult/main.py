from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split



#================================= LOAD Data =================================

URL = "./adult.data.csv"
df  = pd.read_csv(URL)

train, test = train_test_split(df, test_size=0.2)
train, val  = train_test_split(train, test_size=0.2)

#================================= LOAD Data =================================





#=============================== Aanalysis phase of the Data===================


print(len(train), "training length")
print(len(val),   "validation length")
print(len(test),  "testing length")

print(train.head())

def df_to_dataset(df, shuffle=True, batch_size=32):
  df = df.copy()
  labels = df.pop('good_salary')
  ds = tf.data.Dataset.from_tensor_slices((dict(df) , labels))
  if shuffle:
    ds.shuffle(buffer_size=len(df))
  ds = ds.batch(batch_size)
  return ds


batch_size=32
train_ds = df_to_dataset(train, shuffle=True , batch_size=batch_size)
val_ds   = df_to_dataset(val  , shuffle=False, batch_size=batch_size)
test_ds  = df_to_dataset(test , shuffle=False, batch_size=batch_size)

example_batch = next(iter(train_ds))[0]

def demo(feature_column):
  feature_layer = layers.DenseFeatures(feature_column)
  print(feature_layer(example_batch).numpy())


for tr_batch, _ in train_ds.take(1):
  print("keys-->>", list(tr_batch.keys()))

#=============================== Aanalysis phase of the Data===================


#===============================FEATURE Engineering===============================

feature_columns = []
numeric_headers = ['age', 'education_num', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']
for header in numeric_headers:
  feature_columns.append(feature_column.numeric_column(header))

# embedding
# native_country
country = feature_column.categorical_column_with_vocabulary_list('native_country', df['native_country'].unique())
country_embedding = feature_column.embedding_column(country, dimension=40)
feature_columns.append(country_embedding)
#demo(country_embedding)

occupation = feature_column.categorical_column_with_vocabulary_list('occupation', df['occupation'].unique())
occupation_embedding = feature_column.embedding_column(occupation, dimension=10)
feature_columns.append(occupation_embedding)
#demo(occupation_embedding)

education = feature_column.categorical_column_with_vocabulary_list('education', df['education'].unique())
education_embedding = feature_column.embedding_column(education, dimension=20)
feature_columns.append(education_embedding)

print("*****")

#indicator cols
relationship = feature_column.categorical_column_with_vocabulary_list('relationship', df['relationship'].unique()) 
relationship_one_hot = feature_column.indicator_column(relationship)
feature_columns.append(relationship_one_hot)

#sex
sex = feature_column.categorical_column_with_vocabulary_list('sex', df['sex'].unique())
sex_one_hot = feature_column.indicator_column(sex)
feature_columns.append(sex_one_hot)

#race
race = feature_column.categorical_column_with_vocabulary_list('race', df['race'].unique())
race_one_hot = feature_column.indicator_column(race)
feature_columns.append(race_one_hot)

#workclass
workclass = feature_column.categorical_column_with_vocabulary_list('workclass', df['workclass'].unique())
workclass_one_hot = feature_column.indicator_column(workclass)
feature_columns.append(workclass_one_hot)

#bucketize cols
age = feature_column.numeric_column('age')
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

hours_per_week = feature_column.numeric_column('hours_per_week')
hours_per_week_bucketize = feature_column.bucketized_column(hours_per_week, boundaries=[15, 20, 25, 30, 35, 40, 45, 50, 60])
feature_columns.append(hours_per_week_bucketize)

#hashed bucket
martial_status_hashed = feature_column.categorical_column_with_hash_bucket(
                  'martial_status', hash_bucket_size=1000)
martial_status_one_hot = feature_column.indicator_column(martial_status_hashed)
feature_columns.append(martial_status_one_hot)

"""
#crossed cols, age and martial_status, age and martial_status and relationship
ms = feature_column.categorical_column_with_hash_bucket('martial_status', df['martial_status'].unique())
crossed_feature = feature_column.crossed_column([age_buckets, ms], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
"""

#===============================FEATURE Engineering===============================



#================================Model========================
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
print("Accuracy---->", accuracy)



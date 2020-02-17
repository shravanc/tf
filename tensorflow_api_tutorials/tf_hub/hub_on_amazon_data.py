from __future__ import absolute_import, division, print_function, unicode_literals


#=================================================LOAD_DATA========================================
import csv

sentences = []
labels    = []


with open("/tmp/data.csv") as fp:
  lines = csv.reader(fp)
  for row in lines:
    sentences.append(row[0])
    labels.append(int(row[1]))


split_percent = 0.6
split_to      = int(split_percent * len(sentences))

train_sentences = sentences[0:split_to]
val_split     = int(split_percent * len(train_sentences))

val_sentences   = train_sentences[val_split:]
train_sentences = train_sentences[0:val_split]
test_sentences  = sentences[split_to:]


train_labels = labels[0:split_to]
val_labels   = train_labels[val_split:]
train_labels = train_labels[0:val_split]
test_labels  = labels[split_to:]
#=================================================LOAD_DATA========================================


#=================================================DATA_Prep=======================================

def convert_to_dfts(df, labels, shuffle=True, batch_size=32):
  ds = tf.data.Dataset.from_tensor_slices(
      (
      tf.cast(df, tf.string),
      tf.cast(labels, tf.int32)
      )
  )
  return ds


#Creating Dataset for training, validation and testing datatframes.
batch_size=32
train_ds = convert_to_dfts(train_sentences, train_labels, shuffle=True , batch_size=batch_size)
val_ds   = convert_to_dfts(val_sentences  , val_labels  , shuffle=False, batch_size=batch_size)
test_ds  = convert_to_dfts(test_sentences , test_labels , shuffle=False, batch_size=batch_size)
#=================================================DATA_Prep=======================================


#================================================Build and Evaluate Model========================
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

#Embedding Layer
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)


#Building Model
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()


# Compiling
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(train_ds.shuffle(100).batch(32),
                    epochs=100,
                    validation_data=val_ds.batch(32),
                    )

# Evaluation
results = model.evaluate(test_ds.batch(32), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))

#================================================Build and Evaluate Model========================

import tensorflow as tf
import numpy as np
import os
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

url = 'https://raw.githubusercontent.com/shravanc/datasets/master/ham_spam_text_classification.csv'
df = pd.read_csv(url)
mapper = {'ham': 0, 'spam': 1}
df['Category'] = df['Category'].replace(mapper)

train_data = df.sample(frac=0.8, random_state=0)
valid_data = df.drop(train_data.index)

train_labels = train_data.pop('Category')
valid_labels = valid_data.pop('Category')

train_data = train_data['Message'].values
valid_data = valid_data['Message'].values

train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)

vocab_size = 10000
oov_token = '<OOV>'
padding = 'post'
trunc_type = 'post'
max_len = 120
embed_dim = 16


tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    oov_token=oov_token
)

tokenizer.fit_on_texts(train_data)
train_seq = tokenizer.texts_to_sequences(train_data)
train_pad = tf.keras.preprocessing.sequence.pad_sequences(
    train_seq,
    truncating=trunc_type,
    padding=padding,
    maxlen=max_len
)

valid_seq = tokenizer.texts_to_sequences(valid_data)
valid_pad = tf.keras.preprocessing.sequence.pad_sequences(
    valid_seq,
    maxlen=max_len,
    truncating=trunc_type,
    padding=padding
)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_len),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu', return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='relu')),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

print(len(train_labels))
print(len(valid_labels))

print(len(train_pad))
print(len(valid_pad))

history = model.fit(
    train_pad,
    train_labels,
    epochs=10,
    validation_data=(valid_pad, valid_labels)
)

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

imdb, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)

train_data, valid_data = imdb['train'], imdb['test']

train_sentences = []
train_labels = []

for s, l in train_data:
    train_sentences.append(str(s.numpy()))
    train_labels.append(int(l.numpy()))

valid_sentences = []
valid_labels = []

for s, l in valid_data:
    valid_sentences.append(str(s.numpy()))
    valid_labels.append(int(l.numpy()))

train_labels = np.array(train_labels)
valid_labels = np.array(valid_labels)

vocab_size = 1000
oov_token = '<OOV>'
trun_type = 'post'
padding = 'post'
max_len = 120
embed_dim = 16

tokenizer = tf.keras.preprocessing.text.Tokenizer(
    num_words=vocab_size,
    oov_token=oov_token
)
tokenizer.fit_on_texts(train_sentences)
tokenizer.texts_to_sequences(train_sentences)

train_seq = tokenizer.texts_to_sequences(train_sentences)
train_pad = tf.keras.preprocessing.sequence.pad_sequences(
    train_seq,
    truncating=trun_type,
    padding=padding,
    maxlen=max_len
)

valid_seq = tokenizer.texts_to_sequences(valid_sentences)
valid_pad = tf.keras.preprocessing.sequence.pad_sequences(
    valid_seq,
    truncating=trun_type,
    padding=padding,
    maxlen=max_len
)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_len),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=['accuracy']
)

history = model.fit(
    train_pad,
    train_labels,
    epochs=10,
    validation_data=(valid_pad, valid_labels)
)
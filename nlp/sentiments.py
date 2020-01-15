import csv
import numpy as np

words  = []
labels = []
with open("./data/neg.csv") as f:
  lines = csv.reader(f)
  for row in lines:
    words.append(row[0])
    labels.append(int(row[1]))

with open("./data/pos.csv") as f:
  lines = csv.reader(f)
  for row in lines:
    words.append(row[0])
    labels.append(int(row[1]))


split_percent = 0.9 # 90% to 10%
split_to      = int(split_percent * len(words))

tr_words = words[0:split_to]
te_words = words[split_to:]

tr_labels = np.array(labels[0:split_to])
te_labels = np.array(labels[split_to:])


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size  = 10000
oov_tok     = "<OOV>"
max_len     = 5
embed_dim   = 16
trunc_type  = "post"


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(tr_words)
word_index = tokenizer.word_index

seq = tokenizer.texts_to_sequences(tr_words)
pad_seq = pad_sequences(seq, maxlen=max_len, truncating=trunc_type)

te_seq = tokenizer.texts_to_sequences(te_words)
te_pad_seq = pad_sequences(te_seq, maxlen=max_len, truncating=trunc_type)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_len),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

num_epochs = 10
model.fit(pad_seq, tr_labels, epochs=num_epochs, validation_data=(te_pad_seq, te_labels))

e =model.layers[0]
weights = e.get_weights()[0]


reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

import io

out_v = io.open('./data/kan_vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('./data/kan_meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, len(word_index)):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()


"""
colab link for the same:
https://colab.research.google.com/drive/1g7OPNFtW-B5bn5Q1seKaZd9bOHgatOWu
"""


training_sentences = [
    "I love my dog!",
    "I love my cat",
    "you love my dog!",
    "I love stray dog too, because I had one",
    "I hate the dogs. They bite",
    "dogs are dangerous",
    "dogs are dangerously bad animals",
    "dogs are adorable, the are lovely",
    "dogs are irritating"
]
# 1 -> pos
# 0 -> neg
train_labels = [1, 1, 1, 1, 0,0,0,1,1]

testing_sentences = [
    "I really love my dog",
    "my cat is so lovable",
    "cats are dangerous and dogs bite",
    "cat is a bad animal"
]
test_labels = [1, 1, 0, 0]

import numpy as np
training_labels_final = np.array(train_labels)
testing_labels_final  = np.array(test_labels)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size  = 1000
max_len     = 20
oov_tok     = "<OOV>"
embed_dim   = 4
trunc_type  = "post"

tokenizer   = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index  = tokenizer.word_index

sequences   = tokenizer.texts_to_sequences(training_sentences)
padded      = pad_sequences(sequences, maxlen=max_len, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded    = pad_sequences(testing_sequences, maxlen=max_len, truncating=trunc_type)

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_len),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#model.summary()

num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

e = model.layers[0]
weights = e.get_weights()[0]

import io

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
print(reverse_word_index)

out_v = io.open('./exp_vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('./exp_meta.tsv', 'w', encoding='utf-8')
for word_num in range(1,len(word_index)):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()


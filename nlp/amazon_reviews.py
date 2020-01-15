import csv

sentences = []
labels    = []
#with open("./data/amazon_cells_labelled.csv") as f:
with open("./data/data.csv") as f:
  spamreader = csv.reader(f, delimiter=',')
  for row in spamreader:
    sentences.append(row[0])
    labels.append(int(row[1]))


training_sentences  = sentences[0:800]
testing_sentences   = sentences[800:]

training_labels     = labels[0:800]
testing_labels      = labels[800:]

import numpy as np
training_labels_final = np.array(training_labels)
testing_labels_final  = np.array(testing_labels)



import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size  = 10000
max_len     = 120
oov_tok     = "<OOV>"
embed_dim   = 16
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
#print(reverse_word_index)

out_v = io.open('./amazon_vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('./amazon_meta.tsv', 'w', encoding='utf-8')
for word_num in range(1,len(word_index)):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()


pred_sentences = ["it is a bad product"]
tokenized_sentences = tokenizer.texts_to_sequences(pred_sentences)
pad_seqe            = pad_sequences(tokenized_sentences, maxlen=max_len, truncating=trunc_type)

print(pad_seqe)
predictions = model.predict_classes(pad_seqe)
print(predictions)

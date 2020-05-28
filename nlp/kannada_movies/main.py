import csv
import numpy as np
import os



#base_path = "/home/shravan/python_programs/generate_kannada_movie_reviews/modified_data/"

base_path = "./modified_data"

def load_dataset(path):
  reviews = []
  labels = []
  all_files = os.listdir( path )

  for t_file in all_files:
    abs_path = os.path.join(path, t_file)
    with open(abs_path) as fp:
      lines = csv.reader(fp, delimiter='#')
      for row in lines:
        reviews.append(row[0])
        labels.append(int(row[1]))
  
  return (reviews, np.array(labels))

tr_reviews, tr_labels = load_dataset( os.path.join(base_path, 'train') )
te_reviews, te_labels = load_dataset( os.path.join(base_path, 'test') )

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

vocab_size  = 1000000
oov_tok     = "<OOV>"
max_len     = 5
embed_dim   = 16
trunc_type  = "post"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(tr_reviews)
word_index = tokenizer.word_index

seq = tokenizer.texts_to_sequences(tr_reviews)
pad_seq = pad_sequences(seq, maxlen=max_len, truncating=trunc_type)

te_seq  = tokenizer.texts_to_sequences(te_reviews)
te_pad_seq = pad_sequences(te_seq, maxlen=max_len, truncating=trunc_type)

model1 = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, embed_dim, input_length=max_len),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(6, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

model = tf.keras.Sequential([
  tf.keras.layers.Embedding(vocab_size, embed_dim),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), return_sequences=True),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
  tf.keras.layers.Dense(6, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

num_epochs = 5 #5 #5 #5 #5 #10
model.fit(pad_seq, tr_labels, epochs=num_epochs, validation_data=(te_pad_seq, te_labels))

e = model.layers[0]
weights = e.get_weights()[0]

reverse_word_index = dict([ (value, key) for (key, value) in word_index.items() ])

import io

out_v = io.open('./data/kan_vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('./data/kan_meta.tsv', 'w', encoding='utf-8')
print(len(word_index))
for word_num in range(1, len(word_index)):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()



pred_sentences = ["ಒಳ್ಳೆಯ ಚಲನಚಿತ್ರ", "ಕೆಟ್ಟ ಚಲನಚಿತ್ರ"]
tokenized_sentences = tokenizer.texts_to_sequences(pred_sentences)
pad_seqe            = pad_sequences(tokenized_sentences, maxlen=max_len, truncating=trunc_type)

print(pad_seqe)
predictions = model.predict_classes(pad_seqe)
print(predictions)


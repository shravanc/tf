"""
colab link:
https://colab.research.google.com/drive/1g0UI66Jbyymvv_-QfVco-6JlW4Cs4ZAV
"""


#Step 1
#load the data

import json

with open("./data/sarcasm.json") as f:
    datastore = json.load(f)

"""
{'article_link': 'https://www.huffingtonpost.com/entry/versace-black-code_us_5861fbefe4b0de3a08f600d5', 'headline': "former versace store clerk sues over secret 'black code' for minority shoppers", 'is_sarcastic': 0}

"""
sentences = []
labels    = []

for sentence in datastore:
    sentences.append(sentence["headline"])
    labels.append(sentence["is_sarcastic"])

print(sentences[0])
# Step 2
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

#print(word_index)
sample_text = ["hey do you love dogs"]

seq = tokenizer.texts_to_sequences(sample_text)
print(seq)

padded = pad_sequences(seq)
print(padded)

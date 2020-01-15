"""
colab link for the same:
https://colab.research.google.com/drive/1g7OPNFtW-B5bn5Q1seKaZd9bOHgatOWu
"""


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "I love my dog!",
    "I love my cat",
    "you love my dog!",
    "I love stray dog too, because I had one"
]
sentences = [
    "ಪೂರ್ವದ ಹಳಗನ್ನಡ  ಅನಿಶ್ಚಿತ ಕಾಲಘಟ್ಟದಿಂದ ೭ನೇಯ ಶತಮಾನದವರೆಗೆ",
    "ಹಳಗನ್ನಡ  ೭ರಿಂದ ೧೨ನೆಯ ಶತಮಾನದವರೆಗೆ"
]
new_sentences = [
    "I really love my dog",
    "my cat is so lovable"
]

#Tokenization
"""
oov_token fills in the unknown word from the training data. Default it will ignore
"""
#tokenizer = Tokenizer(num_words = 100)
tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index


#Text Sequences
sequences = tokenizer.texts_to_sequences(sentences)
#sequences = tokenizer.texts_to_sequences(new_sentences)


# Padding
"""
Default behavior is 
1. Padding for smaller sentences at the start.......................Override with --> padding     ="post"
2. Truncation for the longer sentences is done at the start.........Override with --> truncating  ="post"
3. max_len is the length of the longest sentence....................Override with --> maxlen      =5
"""
padded    = pad_sequences(sequences)
#padded    = pad_sequences(sequences, padding="post")
#padded    = pad_sequences(sequences, padding="post", maxlen=5)
#padded    = pad_sequences(sequences, padding="post", truncating="post", maxlen=5)




print(word_index)
print(sequences)
print(padded)

import unicodedata
import re
import os
import time
import io

import tensorflow as tf


# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  #w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.rstrip().strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]

def create_dataset(path, num_examples ):
  lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

  #print(lines[:num_examples])

  final = []
  for l in lines: #[:num_examples]:
    word_pairs = []
    words =l.split('\t')
    word_pairs.append(preprocess_sentence(words[0]))
    word_pairs.append(preprocess_sentence(words[1]))
    final.append(word_pairs)


  return zip(*final)

def create_new_dataset(en_path, hi_path):
  en_lines = io.open(en_path, encoding='UTF-8').read().strip().split('\n')
  hi_lines = io.open(hi_path, encoding='UTF-8').read().strip().split('\n')


  final = []
  for i, l in enumerate(en_lines):
    word_pairs = []
    word_pairs.append(preprocess_sentence(en_lines[i]))
    word_pairs.append(preprocess_sentence(hi_lines[i]))
    final.append(word_pairs)

  return zip(*final)


def max_length(tensor):
  return max(len(t) for t in tensor)

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
  # creating cleaned input, output pairs
  targ_lang, inp_lang = create_dataset(path, num_examples)

  input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
  target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def convert(lang, tensor):
  for t in tensor:
    if t!=0:
      print ("%d ----> %s" % (t, lang.index_word[t]))



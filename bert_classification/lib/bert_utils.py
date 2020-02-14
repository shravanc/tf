import os
import re
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

def load_directory_data(directory):
  data = {}
  data['sentence']  = []
  data['sentiment'] = []

  for file_path in os.listdir(directory):
    with open( os.path.join(directory, file_path), "r") as f:
      data['sentence'].append(f.read())
      data['sentiment'].append(re.match("\d+_(\d+)\.txt", file_path).group(1))
  return pd.DataFrame.from_dict(data)


def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, 'pos'))
  neg_df = load_directory_data(os.path.join(directory, 'neg'))

  pos_df['polarity'] = 1
  neg_df['polarity'] = 0

  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


def download_and_load_datasets(force_download=False):
  if force_download:
    dataset = tf.keras.utils.get_file(
              fname="aclImb.tar.gz",
              origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
              extract=True
              )

  train_path = "/home/shravan/tf/local_datasets/aclImdb/train"
  test_path  = "/home/shravan/tf/local_datasets/aclImdb/test"
  train_df   = load_dataset(train_path)
  test_df    = load_dataset(test_path)

  return train_df, test_df


BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature='tokenization_info', as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info['vocab_file'],
                                            tokenization_info['do_lower_case']])

  return bert.tokenization.FullTokenizer(
          vocab_file=vocab_file, do_lower_case=do_lower_case)


label_list = [0, 1]
MAX_SEQ_LENGTH = 128

def get_prediction(in_sentences, tokenizer, estimator):
  labels = ['Negative', 'Positive']
  input_examples = [run_classifier.InputExample(guid="", text_a=x, text_b=None, label=0) for x in in_sentences]
  input_features = run_classifier.convert_examples_to_features(input_examples, label_list, MAX_SEQ_LENGTH, tokenizer)
  predict_input_fn = run_classifier.input_fn_builder(features=input_features, seq_length=MAX_SEQ_LENGTH, is_training=False, drop_remainder=False)
  predictions = estimator.predict(predict_input_fn)
  return [(sentence, prediction['probabilities'], labels[prediction['labels']]) for sentence, prediction in zip(in_sentences, predictions)]



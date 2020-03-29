import os
import math
import datetime


import numpy as np
import tensorflow as tf


import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

bert_abs_path = '/home/shravan/python_programs/bert_leraning/data/'
bert_model_name = 'multi_cased_L-12_H-768_A-12'

bert_ckpt_dir = os.path.join(bert_abs_path, bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")


tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, 'vocab.txt'))


#model_path = "/home/shravan/tf/tf/bert_tf2/content/bert_movie_model"
model_path = "/home/shravan/tf/tf/bert_tf2/bert_model"
model = tf.keras.models.load_model(model_path)
#print(model.summary())







sentences = ['ಒಳ್ಳೆಯ ಚಲನಚಿತ್ರ', 'ಕೆಟ್ಟ ನಟನೆ']
classes   = [0, 1]
pred_tokens = map(tokenizer.tokenize, sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(128-len(tids)),pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

predictions = model.predict(pred_token_ids).argmax(axis=-1)

for text, label in zip(sentences, predictions):
  print("text:", text, "\nintent:", classes[label])
  print()


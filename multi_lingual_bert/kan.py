#reference
# https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1
# https://www.tensorflow.org/hub
# https://github.com/tensorflow/models/blob/master/official/nlp/bert/run_classifier.py
# https://colab.research.google.com/drive/1hMLd5-r82FrnFnBub-B-fVW78Px4KPX1#scrollTo=tU2OpvYrRFNf&forceEdit=true&sandboxMode=true



import tensorflow_hub as hub
import tensorflow as tf
import bert

FullTokenizer = bert.bert_tokenization.FullTokenizer
from tensorflow.keras.models import Model

import math


max_seq_length = 128

def get_inputs(shape, dtype, name):
  return tf.keras.layers.Input(
              shape=shape,
              dtype=dtype,
              name=name,
  )

input_word_ids = get_inputs( (max_seq_length,), tf.int32, "input_word_ids") 
input_mask     = get_inputs( (max_seq_length,), tf.int32, "input_mask")
segment_ids    = get_inputs( (max_seq_length,), tf.int32, "segment_ids")


#BERT_MODEL = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
BERT_MODEL = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
BERT_MODEL = "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(BERT_MODEL, trainable=True)
pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=[pooled_output, sequence_output])


def get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))


def get_segments(tokens, max_seq_length):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))


def get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids






vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = FullTokenizer(vocab_file, do_lower_case)

s = "This is a nice sentence."
s = "ಶುಭ ದಿನ"
stokens = tokenizer.tokenize(s)

stokens = ["[CLS]"] + stokens + ["[SEP]"]

input_ids = get_ids(stokens, tokenizer, max_seq_length)
input_masks = get_masks(stokens, max_seq_length)
input_segments = get_segments(stokens, max_seq_length)



print(stokens)
print(input_ids)
print(input_masks)
print(input_segments)


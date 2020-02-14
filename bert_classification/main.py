#=================================================IMPORTING========================================================

from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf

import tensorflow_hub as hub
from datetime import datetime


import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

print("---1---")

OUTPUT_DIR = '/home/shravan/tf/saved_model'

from tensorflow import keras
import re
import os
#=================================================IMPORTING========================================================



#=================================================LOADING-&-PREPROCESSING-DATASET==================================

from lib.bert_utils import load_directory_data, load_dataset, download_and_load_datasets

force_download = False
train, test = download_and_load_datasets(force_download)


train = train.sample(5000)
test  = test.sample(5000)
print(train.head())


print(train.columns)

DATA_COLUMN   = 'sentence'
LABEL_COLUMN  = 'polarity'

label_list = [0, 1]



train_InputExamples = train.apply(lambda x: bert.run_classifier
                                                .InputExample(
                                                  guid=None,
                                                  text_a=x[DATA_COLUMN],
                                                  text_b=None,
                                                  label=x[LABEL_COLUMN]
                                                 ), axis=1)

test_InputExamples  = test.apply(lambda x: bert.run_classifier
                                               .InputExample(
                                                  guid=None,
                                                  text_a=x[DATA_COLUMN],
                                                  text_b=None,
                                                  label=x[LABEL_COLUMN]
                                                 ), axis=1)



BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

from lib.bert_utils import create_tokenizer_from_hub_module
tokenizer = create_tokenizer_from_hub_module()

print(tokenizer.tokenize("This here's an example of using the BERT tokenizer"))


# We'll set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 128
# Convert our train and test features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)


BATCH_SIZE              = 32
LEARNING_RATE           = 0.00001 #2e-5
NUM_TRAIN_EPOCHS        = 1.0
WARMUP_PROPORTION       = 0.1
SAVE_CHECKPOINTS_STEPS  = 100
SAVE_SUMMARY_STEPS      = 10


num_train_steps   = int(len(train_features)/BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps  = int(num_train_steps * WARMUP_PROPORTION)
print("Number_training_steps--->", num_train_steps)
print("Number of warmup steps-->", num_warmup_steps)
#=================================================LOADING-&-PREPROCESSING-DATASET=================================



#=================================================ESTIMATOR-INITIALIZATION========================================

run_config = tf.estimator.RunConfig(
    model_dir               = OUTPUT_DIR,
    save_summary_steps      = SAVE_SUMMARY_STEPS,
    save_checkpoints_steps  = SAVE_CHECKPOINTS_STEPS
)

from lib.estimator_function import model_fn_builder
model_fn = model_fn_builder(
    num_labels        = len(label_list),
    learning_rate     = LEARNING_RATE,
    num_train_steps   = num_train_steps,
    num_warmup_steps  = num_warmup_steps
)

estimator = tf.estimator.Estimator(
    model_fn  = model_fn,
    config    = run_config,
    params    = {"batch_size": BATCH_SIZE}

)

print("------------------", estimator)

train_input_fn = bert.run_classifier.input_fn_builder(
    features = train_features,
    seq_length = MAX_SEQ_LENGTH,
    is_training=True,
    drop_remainder=False
)

#=================================================ESTIMATOR-INITIALIZATION========================================



#=================================================MODEL-EVALUATION================================================
#===========Training===========
print(f'Beginning Training!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("Training took time-->", datetime.now() - current_time)
#===========Training===========


"""
test_input_fn = run_classifier.input_fn_builder(
    features=test_features,
    seq_length=MAX_SEQ_LENGTH,
    is_training=False,
    drop_remainder=False
)

print(estimator.evaluate(input_fn=test_input_fn,steps=None))


from lib.bert_utils import get_prediction

pred_sentences = [
  "That movie was absolutely awful",
  "The acting was a bit lacking",
  "The film was creative and surprising",
  "Absolutely fantastic!"
]

predictions = get_prediction(pred_sentences, tokenizer, estimator)
print(predictions)

"""
#=================================================MODEL-EVALUATION================================================

from lib.model import create_model
import tensorflow as tf

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from datetime import datetime


def model_fn_builder(num_labels, learning_rate, num_train_steps, num_warmup_steps):
  def model_fn(features, labels, mode, params):
    print("1--->", datetime.now())
    input_ids   = features['input_ids']
    input_mask  = features['input_mask']
    segment_ids = features['segment_ids']
    label_ids   = features['label_ids']

    is_predicting = (mode==tf.estimator.ModeKeys.PREDICT)

    if not is_predicting:
      print("2--->", datetime.now())
      
      (loss, predicted_labels, log_probs) = create_model(is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
      print("2.5--->", datetime.now())

      train_op = bert.optimization.create_optimizer(
          loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

      print("2.75--->", datetime.now())

      accuracy     = tf.metrics.accuracy(label_ids, predicted_labels)
      eval_metrics = {'eval_accuracy': accuracy}
      print("3--->", datetime.now())
     
     
     
      if mode == tf.estimator.ModeKeys.TRAIN:
        print("4--->", datetime.now())
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          train_op=train_op)
      else:
        return tf.estimator.EstimatorSpec(mode=mode,
          loss=loss,
          eval_metric_ops=eval_metrics)


    else:
      print("***here***")
      (predicted_labels, log_probs) = create_model(
        is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)
 
      predictions = {
        'probabilities': log_probs,
        'labels': predicted_labels
      }
      return tf.estimator.EstimatorSpec(mode, predictions=predictions)


  return model_fn




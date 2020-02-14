import tensorflow as tf
import tensorflow_hub as hub
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from datetime import datetime

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def get_bert_inputs(input_ids, input_mask, segment_ids):
  return dict(
    input_ids   = input_ids,
    input_mask  = input_mask,
    segment_ids = segment_ids
  )

def create_model(is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):

  print("m1------->", datetime.now())
  bert_module   = hub.Module(BERT_MODEL_HUB, trainable=True)
  bert_inputs   = get_bert_inputs(input_ids, input_mask, segment_ids)
  bert_outputs  = bert_module(inputs=bert_inputs,signature='tokens', as_dict=True)


  output_layer = bert_outputs['pooled_output']

  hidden_size = output_layer.shape[-1].value

  print("***********SHAPE***************", [num_labels, hidden_size])

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size], initializer=tf.truncated_normal_initializer(stddev=0.02)
  )

  output_bias = tf.get_variable('output_bias', [num_labels], initializer=tf.zeros_initializer())

  print("m2------->", datetime.now())
  with tf.variable_scope('loss'):
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    # Dropout helps prevent overfitting
    logits    = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits    = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    #Convert labels into one-hot encoding
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    print("m3------->", datetime.now())

    # If we are predicting, we want predicted lables and the probabilities
    if is_predicting:
      return (predicted_labels, log_probs)

    # If we are train/eval, compute loss between predicted and actual label
    print("m4------->", datetime.now())
    per_example_loss = -tf.reduce_sum(one_hot_labels*log_probs,axis=-1)
    print("m5------->", datetime.now())
    loss=tf.reduce_mean(per_example_loss)
    print("m6------->", datetime.now())
    return (loss, predicted_labels, log_probs)




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




#https://www.tensorflow.org/tutorials/load_data/tfrecord

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import IPython.display as display

def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int_64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



print(_bytes_feature(b'hello'))
print(_bytes_feature(u'test_byte'.encode('utf-8')))

print(_float_feature(np.exp(1)))

print(_int_64_feature(True))
print(_int_64_feature(1))


#SerializeToString
feature = _float_feature(np.exp(1))
print(feature.SerializeToString())


n_observations = int(1e4)
feature0 = np.random.choice([False, True], n_observations)
print("feature0-->", feature0)

feature1 = np.random.randint(0, 5, n_observations)
print("feature1-->", feature1)

strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]
print("feature2-->", feature2)

feature3 = np.random.randn(n_observations)
print("feature3-->", feature3)


def serialize_example(feature0, feature1, feature2, feature3):
  feature = {
    'feature0': _int_64_feature(feature0),
    'feature1': _int_64_feature(feature1),
    'feature2': _bytes_feature(feature2),
    'feature3': _float_feature(feature3)
  }

  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()


example_observation = []
serialized_example = serialize_example(False, 4, b'goat', 0.9876)
print(serialized_example)


example_proto = tf.train.Example.FromString(serialized_example)
print(example_proto)


#TFRrecord

print(feature1)
print(tf.data.Dataset.from_tensor_slices(feature1))


features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
print(features_dataset)


for f0, f1, f2, f3 in features_dataset.take(1):
  print(f0)
  print(f1)
  print(f2)
  print(f3)


def tf_serialize_example(f0, f1, f2, f3):
  tf_string = tf.py_function(
    serialize_example,
    (f0, f1, f2, f3),
    tf.string)

  return tf.reshape(tf_string, ())

print(tf_serialize_example(f0, f1, f2, f3))


serialized_features_dataset = features_dataset.map(tf_serialize_example) 
print(serialized_features_dataset)


def generator():
  for features in features_dataset:
    yield serialize_example(*features)

serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=())

print(serialized_features_dataset)


#writing TFRecord file
filename = "test.tfrecord"
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)


#Reading TFRead
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)


for raw_record in raw_dataset.take(10):
  print(repr(raw_record))

feature_description= {
    'feature0': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0)
}

def _parse_function(example_proto):
  return tf.io.parse_single_example(example_proto, feature_description)


parsed_dataset = raw_dataset.map(_parse_function)
print(parsed_dataset)

for parsed_record in parsed_dataset.take(10):
  print(repr(parsed_record))



#Writing file
with tf.io.TFRecordWriter(filename) as writer:
  for i in range(n_observations):
    example = serialize_example(feature0[i], feature1[i], feature2[i], feature3[i])
    writer.write(example)


filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)
print(raw_dataset)


for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(example)



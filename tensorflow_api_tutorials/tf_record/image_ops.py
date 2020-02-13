from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import IPython.display as display
from IPython.display import Image



#cat_in_snow = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg', './images')
URL = "./images/320px-Felis_catus-cat_on_snow.jpg"
display.display(Image(filename=URL))


image_string = open(cat_in_snow, 'rb').read()

label = image_labels[cat_in_snow]

def image_example(image_string, label):
  image_shape = tf.image.decode_jpeg(image_string).shape

  feature = {
    'height': _int64_feature(image_shape[0]),
    'width':  _int64_feature(image_shape[1]),
    'depth':  _int64_feature(image_shape[2]),
    'label':  _int64_feature(label),
    'image_raw': _byte_feature(image_string),
  }


  return tf.train.Example(features=tf.train.Features(feature=feature))


for line in str(image_example(image_string, label)).split('\n')[:15]:
  print(line)
print('...')


#writing to file
record_file = 'images.tfrecords'
with tf.io.TFRecordWriter(record_file) as writer:
  for filename, label in image_labels.items():
    image_string = open(filename, 'rb').read()
    tf_example = image_example(image_string, label)
    writer.write(tf_example.SerializeToString())


raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')
image_feature_description = {
  'height': tf.io.FixedLenFeature([], tf.int64),
  'width': tf.io.FixedLenFeature([], tf.int64),
  'depth': tf.io.FixedLenFeature([], tf.int64),
  'label': tf.io.FixedLenFeature([], tf.int64),
  'image_raw': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
  return tf.io.parse_single_example(example_proto, image_feature_description)

parse_image_dataset = raw_image_dataset.map(_parse_image_function)
print(parse_image_dataset)


for image_features in parsed_image_dataset:
  image_raw = image_features['image_raw'].numpy()
  display.display(display.Image(data=image_raw))

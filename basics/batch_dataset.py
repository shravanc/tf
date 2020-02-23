import tensorflow as tf
import numpy as np


num_items = 11
num_list1 = np.arange(num_items)
num_list2 = np.arange(num_items, num_items*2)

dataset1 = [1, 2, 3, 4, 5]
dataset2 = ['a', 'e', 'i', 'o', 'u']
dataset1 = tf.data.Dataset.from_tensor_slices(dataset1)
dataset2 = tf.data.Dataset.from_tensor_slices(dataset2)

zipped_datasets = tf.data.Dataset.zip((dataset1, dataset2))
iterator = tf.compat.v1.data.make_one_shot_iterator(zipped_datasets)
for item in zipped_datasets:
  print(iterator.get_next())

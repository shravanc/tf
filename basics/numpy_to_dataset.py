import tensorflow as tf
import numpy as np


num_items = 11
num_list1 = np.arange(num_items)
num_list2 = np.arange(num_items, num_items*2)

print(num_list1)
print(num_list2)

num_list1_dataset = tf.data.Dataset.from_tensor_slices(num_list1)
print(num_list1_dataset)

iterator = tf.compat.v1.data.make_one_shot_iterator(num_list1_dataset)
for item in num_list1_dataset:
  num = item.numpy()
  print(num)
  print("item--", item)



num_list1_dataset = tf.data.Dataset.from_tensor_slices(num_list1).batch(3, drop_remainder=False)
iterator = tf.compat.v1.data.make_one_shot_iterator(num_list1_dataset)
for item in num_list1_dataset:
  num = iterator.get_next().numpy()
  print(num)

# https://www.tensorflow.org/guide/data
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset

import tensorflow as tf

print(tf.__version__)


#Simplest way to create a dataset
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
print(dataset)

for ele in dataset:
  print(ele)



# process lines from the the files
dataset = tf.data.TextLineDataset(["./sample/file_1.txt", "./sample/file_2.txt"])
for ele in dataset:
  print(ele)


# read all files from a path
dataset = tf.data.Dataset.list_files("./sample/*.txt")
for ele in dataset:
  print(ele)


# transformations
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.map(lambda x: x*2)
dataset = list(dataset.as_numpy_iterator())
print(dataset)


# range
dataset = tf.data.Dataset.range(100)
def dataset_fn(ds):
  return ds.filter(lambda x: x<5)
dataset = dataset.apply(dataset_fn)
dataset = list(dataset.as_numpy_iterator())
print(dataset)


dataset = tf.data.Dataset.from_tensor_slices({'a': ([1, 2], [3, 4]), 'b': ([5, 6])})
print(list(dataset.as_numpy_iterator()))
state = list(dataset.as_numpy_iterator()) == [{'a': (1,2), 'b': 5}, {'a': (2, 4), 'b': 6}]
print(state)



# Batch 
dataset = tf.data.Dataset.range(8)
dataset = dataset.batch(3)
print(list(dataset.as_numpy_iterator()))


# lambda
dataset = tf.data.Dataset.range(3)
dataset = dataset.map(lambda x: x**2)
dataset = dataset.cache()

print(list(dataset.as_numpy_iterator()))

# Concatenate

a = tf.data.Dataset.range(1, 4)
b = tf.data.Dataset.range(4, 8)
ds = a.concatenate(b)

print(list(ds.as_numpy_iterator()))


c = tf.data.Dataset.zip((a, b))
print(list(c.as_numpy_iterator()))

#Below is error, unmatched data types
#a.concatenate(c)


dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
dataset = dataset.enumerate(start=5)
for element in dataset.as_numpy_iterator():
  print(element)


dataset = tf.data.Dataset.from_tensor_slices([(1, 2), (3, 4)])
dataset = dataset.enumerate()
for ele in dataset.as_numpy_iterator():
  print(ele)



# Filter
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4])
dataset = dataset.filter(lambda x: x<3)
for ele in dataset.as_numpy_iterator():
  print(ele)


def filter_fn(x):
  return tf.math.equal(x, 1)

dataset = dataset.filter(filter_fn)
print(list(dataset.as_numpy_iterator()))


# FlatMap
dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [2, 3], [4, 5]])
dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))
print(list(dataset.as_numpy_iterator()))



# Generators
import itertools

def gen():
  for i in itertools.count(1):
    yield(i, [1] * i)

dataset = tf.data.Dataset.from_generator(gen, (tf.int64, tf.int64), (tf.TensorShape([]), tf.TensorShape([None])))

print(list(dataset.take(4).as_numpy_iterator()))


# from_tensor_slices
dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
print(list(dataset.as_numpy_iterator()))

dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})
print(list(dataset.as_numpy_iterator()))


# shuffle
dataset = tf.data.Dataset.range(3)
dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
#dataset = dataset.repeat(4)
print(list(dataset.as_numpy_iterator()))
print(list(dataset.as_numpy_iterator()))
print(list(dataset.as_numpy_iterator()))
print(list(dataset.as_numpy_iterator()))


# Skip
dataset = tf.data.Dataset.range(10)
dataset = dataset.skip(7)
print(list(dataset.as_numpy_iterator()))


# Take
dataset = tf.data.Dataset.range(10)
dataset = dataset.take(3)
print(list(dataset.as_numpy_iterator()))


# Window
dataset = tf.data.Dataset.range(10).window(2, 4, 2, True)
for window in dataset:
  print(list(window.as_numpy_iterator()))



# Apply
dataset = tf.data.Dataset.range(100)
def dataset_fn(ds):
  return ds.filter(lambda x: x<5)
dataset = dataset.apply(dataset_fn)
print(list(dataset.as_numpy_iterator()))
  

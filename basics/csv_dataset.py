import tensorflow as tf

filename = ['size_1000.csv']
record_defaults = [tf.float32] * 2 # two required float columns
dataset = tf.data.experimental.CsvDataset(filename, record_defaults, header=True, select_cols=[1,2])
for item in dataset:
  print(item)

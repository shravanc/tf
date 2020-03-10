import tensorflow as tf
import timeit

print(tf.__version__)

cell = tf.keras.layers.LSTMCell(100)

@tf.function
def fn(input, state):
  return cell(input, state)

input = tf.zeros([100, 100])
print(input)

state = [tf.zeros([100, 100])] * 2

cell(input, state)
fn(input, state)

graph_time = timeit.timeit(lambda: cell(input, state), number=100)
auto_graph_time = timeit.timeit(lambda: fn(input, state), number=100)

print("graph_time:", graph_time)
print("auto_graph_time:", auto_graph_time)

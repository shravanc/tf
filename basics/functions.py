import tensorflow as tf

def f1(x, y):
  return tf.reduce_mean(input_tensor=tf.multiply(x ** 2, 5))


f2 = tf.function(f1)


x = tf.constant([4., -5.])
y = tf.constant([2., 3.])

print(f1(x, y).numpy())
print(f2(x,y).numpy())

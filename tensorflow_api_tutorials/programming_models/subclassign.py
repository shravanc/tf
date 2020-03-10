import tensorflow as tf


class MyLayer(tf.layers.Layer):

  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(MyLayer, self).__init__(**kwargs)


  def build(self, input_shape):
    self.kernel = self.add_weight(name='kernel', shape=(input_shape[1], self.output_dim)), initializer='uniform', trainable=True

  
  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)


model = tf.keras.Sequential([
        MyLayer(20),
        layers.Activation('softmax')
        ])

import tensorflow as tf

def build_model():
  text_input_a = tf.keras.Input(shape=(None,), dtype='int32')

  text_input_b = tf.keras.Input(shape=(None,), dtype='int32')

  shared_embedding = tf.keras.layers.Embedding(1000, 128)
  
  encoded_input_a = shared_embedding(text_input_a)
  encoded_input_b = shared_embedding(text_input_b)

  prediction_a = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction_a')(encoded_input_a)
  prediction_b = tf.keras.layers.Dense(1, activation='sigmoid', name='prediction_b')(encoded_input_b)

  model = tf.keras.Model(inputs=[text_input_a, text_input_b], outputs=[prediction_a, prediction_b])

  tf.keras.utils.plot_model(model, to_file='shared_model.png')

build_model()

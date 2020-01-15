#basic
model = tf.keras.Sequential([
  tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
  tf.keras.Flatten(),
  tf.keras.GlobalAveragePooling1D(),
  tf.keras.layers.Dense(6, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

#Convolution
model = tf.keras.Sequential([
  tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
  tf.keras.Conv1D(128, 5, activation="relu"),
  tf.keras.GlobalAveragePooling1D(),
  tf.keras.layers.Dense(24, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])


#LSTM
model = tf.keras.Sequential([
  tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), return_sequences=True)
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))
  tf.keras.layers.Dense(6, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])


#GRU
model = tf.keras.Sequential([
  tf.keras.layers.Embedding(tokenizer.vocab_size, embedding_dim),
  tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32))
  tf.keras.layers.Dense(6, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

mnist = tf.keras.datasets.fashion_mnist

(train_data, train_labels), (valid_data, valid_labels) = mnist.load_data()

train_data = train_data / 255.
valid_data = valid_data / 255.

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=[28, 28]),
    tf.keras.layers.Dense(1024, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(10, activation=tf.keras.activations.softmax)
])

model.compile(
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    train_labels,
    epochs=10,
    validation_data=(valid_data, valid_labels)
)

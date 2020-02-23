import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x, test_x = train_x/255.0, test_x/255.0
epochs = 10


class MyModel(tf.keras.Model):
  def __init__(self, num_classes=10):
    super(MyModel, self).__init__()
    inputs = tf.keras.Input(shape=(28, 28))
    self.x0 = tf.keras.layers.Flatten()
    self.x1 = tf.keras.layers.Dense(512, activation='relu',name='d1')
    self.x2 = tf.keras.layers.Dropout(0.2)
    self.predictions = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='d2')

  def call(self, inputs):
    x = self.x0(inputs)
    x = self.x1(x)
    x = self.x2(x)
    return self.predictions(x)

model5 = MyModel()

batch_size = 32
buffer_size = 10000

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(32).shuffle(10000)

train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
train_dataset = train_dataset.repeat()

test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size).shuffle(10000)
test_dataset = train_dataset.repeat()



steps_per_epoch = len(train_x)//batch_size # required because of the repeat on the dataset
optimiser = tf.keras.optimizers.Adam()
model5.compile (optimizer= optimiser, loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
model5.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch)

model5.save('/home/shravan/tf/all_models/tens_2')


#loading saved Model
#from tensorflow.keras.models import load_model
#new_model = load_model('./model_name.h5')

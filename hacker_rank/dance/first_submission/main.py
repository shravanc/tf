import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import csv
from tensorflow.keras.preprocessing import image

from tensorflow.keras import layers
from tensorflow.keras import Model

from tensorflow.keras.applications.inception_v3 import InceptionV3

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_path = '/home/shravan/Downloads/dance_dataset/workspace/train/'
test_image_path = '/home/shravan/Downloads/dance_dataset/dataset/test/'

test_path = '/home/shravan/Downloads/dance_dataset/dataset'

image_width = 300
image_height = 300
total_count = 364
batch_size = 32
steps_per_batch = total_count // batch_size
classes = len(os.listdir(train_path))

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255.,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)
train_generator = train_gen.flow_from_directory(
    train_path,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# model = tf.keras.models.Sequential([
#     # Note the input shape is the desired size of the image 150x150 with 3 bytes color
#     # This is the first convolution
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(image_width, image_height, 3)),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     # The second convolution
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     # The third convolution
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     # The fourth convolution
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     # Flatten the results to feed into a DNN
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.5),
#     # 512 neuron hidden layer
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dense(classes, activation='softmax')
# ])
#
# STEPS_PER_EPOCH = total_count // batch_size
# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#     0.0001,
#     decay_steps=STEPS_PER_EPOCH * 1000,
#     decay_rate=1,
#     staircase=False)
#
# model.compile(
#     loss=tf.keras.losses.categorical_crossentropy,
#     optimizer=tf.keras.optimizers.Adam(lr_schedule),
#     metrics=['accuracy', 'mae']
# )
#
# history = model.fit(
#     train_generator,
#     steps_per_epoch=steps_per_batch,
#     epochs=100
# )


# ===Model
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape=(image_width, image_height, 3),
                                include_top=False,
                                weights=None)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)

x = layers.Dense(classes, activation='softmax')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy', 'mae'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=total_count // batch_size,
    epochs=100,
)

plots = ['mae']
for plot in plots:
    metric = history.history[plot]
    # val_metric = history.history[f"val_{plot}"]
    epochs = range(len(metric))

    plt.figure(figsize=(15, 10))
    plt.plot(epochs, metric, label=f"Training {plot}")
    # plt.plot(epochs, val_metric, label=f"Validation {plot}")
    plt.legend()
    plt.title(f"Training and Validation for {plot}")

    plt.show()

csv_file = os.path.join(test_path, 'test.csv')
df = pd.read_csv(csv_file)
print(df.head())

submission = './submission.csv'
csv_sub = open(submission, mode='w')
writer = csv.writer(csv_sub, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
writer.writerow(['Image', 'target'])

dances = ['bharatanatyam', 'kathak', 'kathakali', 'kuchipudi', 'manipuri', 'mohiniyattam', 'odissi', 'sattriya']
for index, row in df.iterrows():
    print("**************>", row.values[0])
    test_image = os.path.join(test_image_path, row.values[0])
    img = image.load_img(test_image, target_size=(image_width, image_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    image_class = np.argmax(classes[0])
    result = [row.values[0], dances[image_class]]
    writer.writerow(result)

csv_sub.close()

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing import image


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

train_path = '/home/shravan/Downloads/dance_dataset/workspace/train'
valid_path = '/home/shravan/Downloads/dance_dataset/workspace/valid'

test_path = '/home/shravan/Downloads/dance_dataset/dataset/test'


def get_counts(path):
    dance_dirs = os.listdir(path)
    total = 0
    for dir in dance_dirs:
        dir = os.path.join(path, dir)
        total += len(os.listdir(dir))

    return total


train_count = get_counts(train_path)
valid_count = get_counts(valid_path)

image_height = 140
image_width = 140
train_batch_size = 10
valid_batch_size = 4
number_of_classes = 8

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=40
)

train_generator = train_gen.flow_from_directory(
    train_path,
    target_size=(image_height, image_width),
    batch_size=train_batch_size,
    class_mode='categorical',
    save_to_dir='./hello',
    save_prefix='aug',
    save_format='png'
)

valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.,

)

valid_generator = valid_gen.flow_from_directory(
    valid_path,
    target_size=(image_height, image_width),
    batch_size=valid_batch_size,
    class_mode='categorical',
)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(number_of_classes, activation='softmax')
])

model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.RMSprop(lr=1e-4),
    metrics=['accuracy', 'mae']
)

model.fit(
    train_generator,
    steps_per_epoch=train_count // train_batch_size,
    epochs=1,
    validation_data=valid_generator,
    validation_steps=valid_count // valid_batch_size
)


for f in os.listdir(test_path):
    filename = os.path.join(test_path, f)
    print(filename)

    img = image.load_img(filename, target_size=(image_width, image_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    classes = model.predict(images, batch_size=10)

    print(classes[0])

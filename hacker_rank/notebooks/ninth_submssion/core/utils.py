import tensorflow as tf
from tensorflow.keras import regularizers
import numpy as np
from sklearn.decomposition import PCA

import csv


# Converting to tf.data
def prepare_dataset(data, labels, conf): #batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(conf.shuffle_buffer)
    dataset = dataset.batch(conf.batch_size).prefetch(1)
    return dataset


def reduce_dimensions(raw_train_data, raw_test_data, conf):
    pca = PCA(n_components=conf.pca_components)
    train_data = pca.fit_transform(raw_train_data)

    test_data = pca.transform(raw_test_data)

    return train_data, test_data


def compile_and_fit(model, conf, train_dataset):
    model.compile(
        loss=conf.loss,
        optimizer=conf.optimizer,
        metrics=conf.metrics
    )
    cb1 = tf.keras.callbacks.EarlyStopping(
        monitor='mse',
        patience=20,
    )
    cb2 = tf.keras.callbacks.TensorBoard(
        log_dir='logs', histogram_freq=0, write_graph=True, write_images=False,
        update_freq='epoch', profile_batch=2, embeddings_freq=0
    )

    history = model.fit(
        train_dataset,
        epochs=conf.epochs,
        callbacks=[cb1, cb2]
    )

    return model, history


def get_forecast(model, data):
    forecast = []
    for index, row in enumerate(data):
        prediction_data = row[np.newaxis]
        prediction = model.predict(prediction_data)
        result = prediction[0, 0]
        forecast.append(result)

    return forecast


def write_to_csv(index, target, filename):
    fp = open(filename, mode='w')
    writer = csv.writer(fp, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Employee_ID', 'Attrition_rate'])
    for index, row in enumerate(index):
        data = [row, target[index]]
        writer.writerow(data)

    fp.close()
    return

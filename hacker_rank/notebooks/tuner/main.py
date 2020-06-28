import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import IPython
import kerastuner as kt
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.decomposition import PCA


train_path = 'https://raw.githubusercontent.com/shravanc/datasets/master/clean_train.csv'
df = pd.read_csv(train_path)
df.pop('Unnamed: 0')

# df = df[:1000]

train_df = df.sample(frac=0.6, random_state=2)
valid_df = df.drop(train_df.index)

TARGET = 'Attrition_rate'
train_labels = train_df.pop(TARGET)
valid_labels = valid_df.pop(TARGET)

# Scale
stats = train_df.describe().transpose()
train_df = (train_df - stats['mean']) / stats['std']
valid_df = (valid_df - stats['mean']) / stats['std']

# Convert to numpy
train_data = train_df.to_numpy()
valid_data = valid_df.to_numpy()

pca = PCA(n_components=16)
train_data = pca.fit_transform(train_data)
valid_data = pca.transform(valid_data)


# Converting to tf.data
def prepare_dataset(data, labels, batch, shuffle_buffer):
    # data = tf.expand_dims(data, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)  # .repeat()
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


# Training and Validation Dataset
batch_size = 32
buffer = 100
STEPS_PER_EPOCH = len(train_df) // batch_size

train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH * 1000,
    decay_rate=1,
    staircase=False)


def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 8)):
        model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                  min_value=32,
                                                  max_value=1028,
                                                  step=32),
                                     activation='sigmoid',
                                     kernel_regularizer=regularizers.l2(hp.Float('learning_rate_'+str(i), 1e-4, 1e-2))
                                     ))
        hp_units = hp.Float('dropout_'+str(i), 0, 0.5, step=0.1, default=0.5)
        model.add(keras.layers.Dropout(hp_units))

        # model.add(tf.keras.layers.BatchNormalization())
    #hp_units = hp.Float('dropout', 0, 0.5, step=0.1, default=0.5)
    # model.add(keras.layers.Dropout(hp_units))

    model.add(keras.layers.Dense(1))


    model.compile(
        optimizer=keras.optimizers.SGD(lr=hp.Float('learning_rate', 1e-4, 1e-2), momentum=0.9),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mse'])
    return model


tuner = kt.Hyperband(build_model,
                     objective='mse',
                     max_epochs=10,
                     directory='test_2',
                     project_name='intro_to_kt')



class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


tuner.search(train_dataset, epochs=10, validation_data=valid_dataset, callbacks=[ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


for i, layer in enumerate(range(best_hps.get('num_layers'))):
    unit = f"units_{i}"
    dp = f"dropout_{i}"
    lr = f"learning_rate_{i}"

    print(f"Layer_{i}-->{best_hps.get(unit)}")
    print(f"DP_{i}-->{best_hps.get(dp)}")
    print(f"LR_{i}-->{best_hps.get(lr)}")


print(f"learning --rate--->{best_hps.get('learning_rate')}")
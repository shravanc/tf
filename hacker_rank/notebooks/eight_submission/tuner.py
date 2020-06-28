import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU
from sklearn.decomposition import PCA
from tensorflow import keras
from tensorflow.keras import regularizers
import kerastuner as kt
import IPython


train_path = '~/Downloads/hacker_rank/Dataset/Train.csv'
raw_train_df = pd.read_csv(train_path)

test_path = '~/Downloads/hacker_rank/Dataset/Test.csv'
raw_test_df = pd.read_csv(test_path)

index_column = 'Employee_ID'
train_index = raw_train_df.pop(index_column)
test_index = raw_test_df.pop(index_column)

# Merging both train and test
df = pd.concat([raw_train_df, raw_test_df], ignore_index=True)

# Categorical column
categorical_columns = ['Gender', 'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess',
                       'Compensation_and_Benefits']
df = pd.get_dummies(df, columns=categorical_columns)

# print(df.isna().sum())
# Imputation for Time_of_service
# Time_of_service is related to Time_since_promotion
ptable = df.pivot_table(
    values='Time_of_service',
    index='Time_since_promotion',
    aggfunc=np.mean
)


def get_element(x):
    index = int(x['Time_since_promotion'])
    return ptable.loc[index].values[0]


df['Time_of_service'].fillna(
    df[df['Time_of_service'].isnull()].apply(get_element, axis=1),
    inplace=True
)

# Imputation for Age
ptable = df.pivot_table(
    values='Age',
    index='Time_since_promotion',
    columns='Time_of_service'
)


def get_element(x):
    index = x['Time_since_promotion']
    columns = x['Time_of_service']
    return ptable.loc[index, columns]


df['Age'].fillna(
    df[df['Age'].isnull()].apply(get_element, axis=1),
    inplace=True
)

# Imputation of unrelated columns
missing_numeric_columns = ['Pay_Scale', 'VAR2', 'VAR4', 'Work_Life_balance']
for col in missing_numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Data Preparation=========================================
split_index = 7000
original_df = df[:7000]
test_df = df.drop(original_df.index)

# Remove traget column
TARGET = 'Attrition_rate'


# Converting to tf.data
def prepare_dataset(data, labels, batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


batch_size = 32
buffer = 1000
epochs = 1000
STEPS_PER_EPOCH = 7000 // batch_size
final_labels = original_df.pop(TARGET)
test_df.pop(TARGET)
pca_components = 10

# Scale
final_df = original_df.copy()
stats = final_df.describe().transpose()
final_df = (final_df - stats['mean']) / stats['std']
test_df = (test_df - stats['mean']) / stats['std']

# Convert to numpy
raw_final_data = final_df.to_numpy()
raw_test_data = test_df.to_numpy()

pca = PCA(n_components=pca_components)
# pca = PCA()
final_data = pca.fit_transform(raw_final_data)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

print(len(explained_variance))

test_data = pca.transform(raw_test_data)
final_dataset = prepare_dataset(final_data, final_labels, batch_size, buffer)

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.0001,
    decay_steps=STEPS_PER_EPOCH * 1000,
    decay_rate=1,
    staircase=False)


def build_model(hp):
    model = tf.keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 10)):
        model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                     min_value=32,
                                                     max_value=1028,
                                                     step=32),
                                        activation='sigmoid',
                                        kernel_regularizer=regularizers.l2(
                                            hp.Float('learning_rate_' + str(i), 1e-4, 1e-2))
                                        ))
        hp_units = hp.Float('dropout_' + str(i), 0, 0.5, step=0.1, default=0.5)
        model.add(tf.keras.layers.Dropout(hp_units))

    model.add(tf.keras.layers.Dense(1))

    opt_lr = hp.Float('optimiser_lr', 1e-6, 1e-2)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(opt_lr),
        loss=tf.keras.losses.Huber(delta=1.0),
        metrics=['mse'])
    return model


tuner = kt.Hyperband(build_model,
                     objective='mse',
                     max_epochs=10,
                     directory='test2',
                     project_name='attrition_rate')


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


tuner.search(final_dataset, epochs=10, callbacks=[ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]


print("--Optimal Number of Layers---->", best_hps.get('num_layers'))
for i, layer in enumerate(range(best_hps.get('num_layers'))):
    unit = f"units_{i}"
    dp = f"dropout_{i}"
    lr = f"learning_rate_{i}"

    print(f"Layer_{i}-->{best_hps.get(unit)}")
    print(f"DP_{i}-->{best_hps.get(dp)}")
    print(f"LR_{i}-->{best_hps.get(lr)}")


print("opt_lr--->", best_hps.get('optimiser_lr'))

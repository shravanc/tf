import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU

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
    # df[col].fillna(df[col].mean(), inplace=True)
    df.pop(col)

split_index = 7000
original_df = df[:7000]
test_df = df.drop(original_df.index)

# ===== Validate =====
train_df = original_df.sample(frac=.85, random_state=2)
valid_df = original_df.drop(train_df.index)

# Remove traget column
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


# Converting to tf.data
def prepare_dataset(data, labels, batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


batch_size = 10
buffer = 1000
epochs = 100000
STEPS_PER_EPOCH = len(train_df) // batch_size

train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)

# activation = tf.keras.activations.relu
activation = tf.keras.activations.tanh
activation = tf.keras.layers.LeakyReLU(alpha=0.1)
activation = tf.keras.activations.sigmoid
# activation = tf.keras.activations.elu
# kernel_regularizer=regularizers.l2(0.001)),
base_model = tf.keras.Sequential([
    # 1
    tf.keras.layers.Dense(352, activation=activation, kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),

    # 2
    tf.keras.layers.Dense(160, activation=activation, kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.2),

    # 3
    tf.keras.layers.Dense(672, activation=activation, kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.1),

    # 4
    tf.keras.layers.Dense(96, activation=activation, kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.3),

    # 5
    tf.keras.layers.Dense(544, activation=activation, kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.1),

    tf.keras.layers.Dense(1)
])


def get_loss():
    return tf.keras.losses.mse


lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.0001,
    decay_steps=STEPS_PER_EPOCH * 1000,
    decay_rate=1,
    staircase=False)


def get_optimizer():
    return tf.keras.optimizers.SGD(lr_schedule)


def get_metrics():
    return ['mse', 'mae']


def get_monitor():
    return 'mse'


model = base_model

model.compile(
    loss=get_loss(),
    optimizer=get_optimizer(),
    metrics=get_metrics()
)

cb = tf.keras.callbacks.EarlyStopping(
    monitor=f"val_{get_monitor()}",
    patience=20,
)

history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset,
    callbacks=[cb]
)

# plots = get_metrics()
# for plot in plots:
#     metric = history.history[plot]
#     val_metric = history.history[f"val_{plot}"]
#     epochs = range(len(metric))
#
#     plt.figure(figsize=(15, 10))
#     plt.plot(epochs, metric, label=f"Training {plot}")
#     plt.plot(epochs, val_metric, label=f"Validation {plot}")
#     plt.legend()
#     plt.title(f"Training and Validation for {plot}")
#
#     plt.show()  # (file_name)
#
# # Forecast
# forecast = []
# for index, row in valid_df.iterrows():
#     prediction_data = row.values[np.newaxis]
#     prediction = model.predict(prediction_data)
#     forecast.append(prediction[0][0])
#
# original_data = valid_labels[-50:]
# forecast_data = forecast[-50:]
# x_axis = range(len(original_data))
#
# plt.figure(figsize=(15, 10))
# plt.plot(x_axis, original_data, label='Original Data')
# plt.plot(x_axis, forecast_data, label='Prediction Data')
# plt.legend()
# plt.show()

# print("------------------->", tf.keras.metrics.mean_absolute_error(original_data, forecast_data).numpy())
# ====================

# ==== Retrain the Model
final_labels = original_df.pop(TARGET)
test_df.pop(TARGET)

# Scale
final_df = original_df.copy()
stats = final_df.describe().transpose()
final_df = (final_df - stats['mean']) / stats['std']
test_df = (test_df - stats['mean']) / stats['std']

# Convert to numpy
final_data = final_df.to_numpy()

final_dataset = prepare_dataset(final_data, final_labels, batch_size, buffer)

model = base_model
model.compile(
    loss=get_loss(),
    optimizer=get_optimizer(),
    metrics=get_metrics()
)

cb = tf.keras.callbacks.EarlyStopping(
    monitor=get_monitor(),
    patience=50,
)

history = model.fit(
    train_dataset,
    epochs=1000,
    callbacks=[cb]
)

print(test_df.head())
final_csv = "./submission.csv"
csv_file = open(final_csv, 'w')
writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
writer.writerow(['Employee_ID', TARGET])

i = 0
for index, row in test_df.iterrows():
    print(i)
    pd = row.values[np.newaxis]
    prediction = model.predict(pd)
    results = [test_index[i], prediction[0][0]]
    writer.writerow(results)
    i += 1

csv_file.close()
# ====================
#
#
# ==== Make Prediction
#
# ====================

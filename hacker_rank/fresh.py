import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow_docs as tfdocs

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
plot_path = '/home/shravan/Desktop/model_plots/'

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

print(df.isna().sum())
split_index = 7000
original_df = df[:7000]
test_df = df.drop(original_df.index)

# original_csv = './clean_train_1.csv'
# test_csv = './clea_test_1.csv'
# original_df.to_csv(original_csv)
# test_df.to_csv(test_csv)

# Model Analysis

# Split the data into train and valid
train_df = original_df.sample(frac=0.85, random_state=2)
valid_df = original_df.drop(train_df.index)

# Target column:
TARGET = 'Attrition_rate'
train_labels = train_df.pop(TARGET)
valid_labels = valid_df.pop(TARGET)

# Scale
stats = train_df.describe().transpose()
norm_train_df = (train_df - stats['mean']) / stats['std']
norm_valid_df = (valid_df - stats['mean']) / stats['std']

# norm_train_df = train_df.copy()
# norm_valid_df = test_df.copy()

# Converting to numpy
train_data = norm_train_df.to_numpy()
valid_data = norm_valid_df.to_numpy()


# Converting to tf.data
def prepare_dataset(data, labels, batch, shuffle_buffer):
    # data = tf.expand_dims(data, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


batch_size = 32
buffer = 1000
epochs = 300
lr_rate = 1e-2
momentum = 0.9
STEPS_PER_EPOCH = len(train_df) // batch_size

train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)

depth_level = "lr_3_1"
# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1028, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1)
])

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH * 1000,
    decay_rate=1,
    staircase=False)

model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.Adam(lr=lr_schedule),
    metrics=['mse', 'mae']
)


def get_callbacks():
    return [
        tfdocs.modeling.EpochDots(),
        tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
        # tf.keras.callbacks.TensorBoard(logdir/name),
    ]


history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset,
    callbacks=get_callbacks(),
    verbose=0
)


plots = ['mse', 'mae']
for plot in plots:
    metric = history.history[plot]
    val_metric = history.history[f"val_{plot}"]
    epochs = range(len(metric))

    plt.figure(figsize=(15, 10))
    plt.plot(epochs, metric, label=f"Training {plot}")
    plt.plot(epochs, val_metric, label=f"Validation {plot}")
    plt.legend()
    plt.title(f"Training and Validation for {plot}")

    file_name = f"depth_levl_{depth_level}_tr_va_{plot}"
    file_name = os.path.join(plot_path, file_name)
    print(file_name)
    plt.savefig(file_name)

# # Forecast
# forecast = []
# for index, row in norm_valid_df.iterrows():
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
# file_name = os.path.join(plot_path, 'forecast')
# plt.savefig(file_name)
#
# final_mae = tf.keras.losses.mean_absolute_error(original_data, forecast_data)
# print(final_mae)
#
# final_mse = tf.keras.losses.mean_squared_error(original_data, forecast_data)
# print(final_mae)

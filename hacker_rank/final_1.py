import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tensorflow as tf
import csv


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load Train Data
url = '/home/shravan/Downloads/hacker_rank/Dataset/Train.csv'
raw_train_df = pd.read_csv(url)
train_index = raw_train_df.pop('Employee_ID')

# Load Test Data
url = '/home/shravan/Downloads/hacker_rank/Dataset/Test.csv'
raw_test_df = pd.read_csv(url)
test_index = raw_test_df.pop('Employee_ID').values

# Concatenating both the data to impute the missing values
df = pd.concat([raw_train_df, raw_test_df], ignore_index=True)

# Categorical Columns:
categorical_columns = ['Gender', 'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess',
                       'Compensation_and_Benefits']
df = pd.get_dummies(df, columns=categorical_columns)

start_col = df.columns[:5]
print(start_col)
print(df.isna().sum().values)
# sns.heatmap(df[start_col].corr())
# plt.show()

# ===================Imputation Begins================
# Imputation for Time_of_service
# Time of service is related to Age and Time_since_promotion
ptable = df.pivot_table(
    values='Time_of_service',
    index='Time_since_promotion',
    aggfunc=np.mean
)
print(ptable)
print(ptable.iloc[0].values[0])


def get_element(x):
    index = int(x['Time_since_promotion'])
    return ptable.iloc[index].values[0]


df['Time_of_service'].fillna(
    df[df['Time_of_service'].isnull()].apply(
        get_element,
        axis=1),
    inplace=True
)

# Imputation for Age
# Age is related to Time_of_Service and Time_since_promotion
ptable = df.pivot_table(
    values='Age',
    index='Time_of_service',
    columns='Time_since_promotion',
    aggfunc=np.mean
)
print(ptable)


def get_element(x):
    index = x['Time_of_service']
    columns = x['Time_since_promotion']
    return ptable.loc[index, columns]


df['Age'].fillna(
    df[df['Age'].isnull()].apply(
        get_element,
        axis=1
    ),
    inplace=True
)

# For numeric column which are not related ro other fields
missing_numeric_columns = ['Pay_Scale', 'VAR2', 'VAR4', 'Work_Life_balance']
for col in missing_numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

print(df.isna().sum())

# ====================Imputation Ends=====================
split_index = 7000
original_df = df[:split_index]
test_df = df.drop(original_df.index)

print(test_df.head())

train_df = original_df.sample(frac=0.85, random_state=2)
valid_df = original_df.drop(train_df.index)

TARGET = 'Attrition_rate'
train_labels = train_df.pop(TARGET).values
valid_labels = valid_df.pop(TARGET).values

# Scale Train and Valid DF
stats = train_df.describe().transpose()
train_df = (train_df - stats['mean']) / stats['std']
valid_df = (valid_df - stats['mean']) / stats['std']
ntest_df = (test_df - stats['mean']) / stats['std']


# Convert to numpy
train_data = train_df.to_numpy()
valid_data = valid_df.to_numpy()


# Convert to tf.data
def prepare_data(data, labels, batch, shuffle_buffer):
    data = tf.expand_dims(data, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


# Prepare dataset for train and valid
batch_size = 32
buffer = 1000
epochs = 20
train_dataset = prepare_data(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_data(valid_data, valid_labels, batch_size, buffer)

# path = './models/'
# model = tf.keras.models.load_model(path)

# Build Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=128, kernel_size=24,
                           strides=1, padding="causal",
                           activation="relu",
                           input_shape=[None, 1]),
    tf.keras.layers.Conv1D(filters=128, kernel_size=24,
                           strides=1, padding="causal",
                           activation="relu"),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(1028, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1),
])

model.compile(
    loss=tf.keras.losses.mse,
    optimizer=tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9),
    metrics=['mse', 'mae']
)

history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=valid_dataset
)

path = './models/'
model.save(path)

plots = ['mse', 'mae']  # , 'loss']
for plot in plots:
    metric = history.history[plot]
    val_metric = history.history[f"val_{plot}"]
    epochs = range(len(metric))

    plt.figure(figsize=(15, 10))
    plt.plot(epochs, metric, label=f"Training {plot}")
    plt.plot(epochs, val_metric, label=f"Validation {plot}")
    plt.legend()
    plt.title(f"Training and Validation for {plot}")

    file_name = f"tr_va_{plot}"
    plt.show()

# Forecasting
forecast = []
i = 0
for index, row in valid_df.iterrows():
    data = row.values
    f = []
    for d in data:
        f.append([d])
    a = np.array(f)[np.newaxis]
    result = model.predict(a)
    i += 1
    # print(f"Expected Result: {valid_labels[i]}, Prediction Result: {result}")
    forecast.append(result[0][0])

original_data = valid_labels[-50:]
forecast_data = forecast[-50:]
x_axis = range(len(original_data))

plt.figure(figsize=(15, 10))
plt.plot(x_axis, original_data, label='Original Data')
plt.plot(x_axis, forecast_data, label='Prediction Data')
plt.legend()
plt.show()

#
# #========================================Final=========================================
# y = original_df.pop(TARGET)
#
#
# # Scale
# f_stats = original_df.describe().transpose()
# original_df = (original_df-f_stats['mean'])/f_stats['std']
# test_df = (test_df-f_stats['mean'])/original_df['std']
#
#
# x = original_df.to_numpy()
# final_dataset = prepare_data(x, y, batch_size, buffer)
#
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv1D(filters=128, kernel_size=24,
#                            strides=1, padding="causal",
#                            activation="relu",
#                            input_shape=[None, 1]),
#     tf.keras.layers.Conv1D(filters=128, kernel_size=24,
#                            strides=1, padding="causal",
#                            activation="relu"),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     tf.keras.layers.Dense(1028, activation="relu"),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.Dropout(0.4),
#     tf.keras.layers.Dense(1),
# ])
#
# model.compile(
#     loss=tf.keras.losses.mse,
#     optimizer=tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9),
#     metrics=['mse', 'mae']
# )
#
# model.fit(
#     final_dataset,
#     epochs=epochs,
# )
#
# csv_file = open('./final.csv', model="w")
# writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#
# i = 0
# for index, row in test_df.iterrows():
#     data = row.values
#     f = []
#     for d in data:
#         f.append([d])
#     a = np.array(f)[np.newaxis]
#     result = model.predict(a)
#
#     csv_data = [train_index[i], result[0][0]]
#     writer.writerow(csv_data)
#     print("csv_data----->", csv_data)
#
#     i += 1
#
#
# csv_file.close()
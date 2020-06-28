import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


# Load Train Data
url = '/home/shravan/Downloads/hacker_rank/Dataset/Train.csv'
# url = '/home/shravan/Downloads/hacker_rank/Dataset/hr_train.csv'
train_df = pd.read_csv(url)
train_index = train_df.pop('Employee_ID')

# Load Test Data
url = '/home/shravan/Downloads/hacker_rank/Dataset/Test.csv'
# url = '/home/shravan/Downloads/hacker_rank/Dataset/hr_test.csv'
test_df = pd.read_csv(url)
test_index = test_df.pop('Employee_ID').values

df = pd.concat([train_df, test_df], ignore_index=True)
print(df.head(10))

# Target
TARGET = 'Attrition_rate'

# Categorical Columns
categorical_columns = ['Gender', 'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess',
                       'Compensation_and_Benefits']

# Numerical Columns
numerical_columns = ['growth_rate', 'Travel_Rate',
                     'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR6', 'VAR7']

# Numeric Yet Categorical Column
numeric_yet_categorical = ['Education_Level', 'Post_Level']

missing_numeric_columns = ['Pay_Scale', 'VAR2', 'VAR4', 'Work_Life_balance']
related_numeric_columns = ['Age', 'Time_of_service', 'Time_since_promotion']


# ==========Age Imputation=========
ptable = df.pivot_table(values='Age',
                        columns='Time_since_promotion',
                        aggfunc=np.median)


def get_element(x):
    return ptable[x['Time_since_promotion']].values[0]


df['Age'].fillna(df[df['Age'].isnull()].apply(get_element, axis=1), inplace=True)


# ==========Time_of_service Imputation=========
ptable = df.pivot_table(
    values="Time_of_service",
    columns='Time_since_promotion',
    index='Age'
)


def get_element(x):
    return ptable.loc[x['Age'], x['Time_since_promotion']]


df['Time_of_service'].fillna(df[df['Time_of_service'].isnull()].apply(get_element, axis=1), inplace=True)


# =========Imputation of unrelated Numeric Columns=

for col in missing_numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)


# Categorical Columns
df = pd.get_dummies(df, columns=categorical_columns)
df = pd.get_dummies(df, columns=numeric_yet_categorical)


print(df.isna().sum())

split_index = 7000
orig_df = df[:split_index]
test_df = df.drop(train_df.index)


# Predictive Analysis
train_df = orig_df.sample(frac=0.85, random_state=0)
valid_df = orig_df.drop(train_df.index)

train_labels = train_df.pop(TARGET)
valid_labels = valid_df.pop(TARGET)

# Scale the features
stats = train_df.describe().transpose()
train_df = (train_df-stats['mean'])/stats['std']
valid_df = (valid_df-stats['mean'])/stats['std']
test_df = (test_df-stats['mean'])/stats['std']


# Converting to numpy
train_data = train_df.to_numpy()
valid_data = valid_df.to_numpy()


# Converting numpy to tf.data
def prepare_dataset(data, labels, batch, shuffle_buffer):
    data = tf.expand_dims(data, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


# Preparing dataset for training and validation
batch_size = 32
buffer = 1000
train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)

# path = './models/'
# model = tf.keras.models.load_model(path)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=128, kernel_size=24,
                           strides=1, padding="causal",
                           activation="relu",
                           input_shape=[None, 1]),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)),
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

lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-3*10**(epoch/20)
)
model.compile(
    loss=tf.keras.losses.mse,
    optimizer=tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9),
    metrics=[ 'mse', 'mae']
)


callback = tf.keras.callbacks.EarlyStopping(monitor='mae', patience=3)
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=valid_dataset,
    #callbacks=[callback]
)

path = './models/'
model.save(path)


plots = ['mse', 'mae'] #, 'loss']
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


# Forecating
forecast = []
for index, row in valid_df.iterrows():
    data = row.values
    f = []
    for d in data:
        f.append([d])
    a = np.array(f)[np.newaxis]
    result = model.predict(a)
    forecast.append(result[0][0])


original_data = valid_labels[-50:]
forecast_data = forecast[-50:]
x_axis = range(len(original_data))


plt.figure(figsize=(15, 10))
plt.plot(x_axis, original_data, label="Original Data")
plt.plot(x_axis, forecast_data, label="Prediction Data")
plt.legend()
plt.title("Forecast Data")
plt.show()


# print(len(test_df))
#
# predictions = []
# i = 0
# for index, row in test_df.iterrows():
#     data = row.values
#     f = []
#     for d in data:
#         f.append([d])
#     a = np.array(f)[np.newaxis]
#     print(a)
#     result = model.predict(a)
#     store = [test_index[i], result[0][0]]
#     i += 1
#     print(store)
#     predictions.append([test_index[index], result[0][0]])
#     break
#
#
# submission_csv = './submission_csv.csv'
# with open(submission_csv, mode='w') as csv_file:
#     writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#     for prediction in predictions:
#         writer.writerow(prediction)
#
#
# print("*********Done********")
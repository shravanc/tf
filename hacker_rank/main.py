import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


url = '/home/shravan/Downloads/hacker_rank/Dataset/Train.csv'
df = pd.read_csv(url)
df = df.drop('Employee_ID', axis=1)
print(df.head())

# Categorical Columns
categorical_columns = ['Gender', 'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess',
                       'Compensation_and_Benefits']

# Numerical Columns
numerical_columns = ['Age', 'Time_of_service', 'Time_since_promotion', 'growth_rate', 'Travel_Rate',
                     'VAR1', 'VAR2', 'VAR3', 'VAR4', 'VAR6', 'VAR7']

# Target
TARGET = 'Attrition_rate'

# Numeric Yet Categorical Column
numeric_yet_categorical = ['Education_Level', 'Post_Level']

missing_numeric_columns = ['Pay_Scale', 'VAR2', 'VAR4', 'Work_Life_balance']

print(df.isna().sum())
print(len(df))

# sns.heatmap(df.corr())
# plt.show()

df = pd.get_dummies(df, columns=categorical_columns)
df = pd.get_dummies(df, columns=numeric_yet_categorical)

for col in missing_numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

ptable = df.pivot_table(values='Age',
                        index='Time_since_promotion',
                        columns='Travel_Rate',
                        aggfunc=np.median)


# Define function to return an element of the pivot table
def get_element(x):
    return ptable.loc[x['Time_since_promotion'], x['Travel_Rate']]


# Replace missing values
df['Age'].fillna(df[df['Age'].isnull()].apply(get_element, axis=1), inplace=True)


ptable = df.pivot_table(values='Time_of_service',
                        index='Age',
                        columns='Time_since_promotion',
                        aggfunc=np.median)


# Define function to return an element of the pivot table
def get_element(x):
    return ptable.loc[x['Age'], x['Time_since_promotion']]


df['Time_of_service'].fillna(df[df['Time_of_service'].isnull()].apply(get_element, axis=1), inplace=True)


# Split the data
train_df = df.sample(frac=0.80, random_state=0)
valid_df = df.drop(train_df.index)

train_labels = train_df.pop(TARGET).values
valid_labels = valid_df.pop(TARGET).values

# Scale
stats = train_df.describe().transpose()
train_df = (train_df-stats['mean'])/stats['std']
valid_df = (valid_df-stats['mean'])/stats['std']

train_data = train_df.to_numpy()
valid_data = valid_df.to_numpy()


def prepare_dataset(data, labels, batch, shuffle_buffer):
    data = tf.expand_dims(data, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


batch_size = 10
buffer = 500
train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)


# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(512, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(1)
# ])

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
    epochs=50,
    validation_data=valid_dataset,
    #callbacks=[callback]
)


# plotting Graphs
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




url = '/home/shravan/Downloads/hacker_rank/Dataset/Test.csv'
test_df = pd.read_csv(url)
ids = test_df.pop('Employee_ID')


test_df = pd.get_dummies(test_df, columns=categorical_columns)
test_df = pd.get_dummies(test_df, columns=numeric_yet_categorical)

for col in missing_numeric_columns:
    test_df[col].fillna(df[col].mean(), inplace=True)

ptable = df.pivot_table(values='Age',
                        index='Time_since_promotion',
                        columns='Travel_Rate',
                        aggfunc=np.median)


# Define function to return an element of the pivot table
def get_element(x):
    return ptable.loc[x['Time_since_promotion'], x['Travel_Rate']]


# Replace missing values
df['Age'].fillna(df[df['Age'].isnull()].apply(get_element, axis=1), inplace=True)


ptable = df.pivot_table(values='Time_of_service',
                        index='Age',
                        columns='Time_since_promotion',
                        aggfunc=np.median)


# Define function to return an element of the pivot table
def get_element(x):
    return ptable.loc[x['Age'], x['Time_since_promotion']]


df['Time_of_service'].fillna(df[df['Time_of_service'].isnull()].apply(get_element, axis=1), inplace=True)

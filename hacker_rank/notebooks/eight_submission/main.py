import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU
from sklearn.decomposition import PCA

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
    # df.pop(col)

sigmoid = tf.keras.activations.sigmoid
activation = sigmoid
model = tf.keras.Sequential([
    # 1
    tf.keras.layers.Dense(96, activation=activation, kernel_regularizer=regularizers.l2(0.003)),
    tf.keras.layers.Dropout(0.5),

    # 2
    tf.keras.layers.Dense(768, activation=activation, kernel_regularizer=regularizers.l2(0.004)),
    tf.keras.layers.Dropout(0.5),

    # 3
    tf.keras.layers.Dense(288, activation=activation, kernel_regularizer=regularizers.l2(0.003)),
    tf.keras.layers.Dropout(0.2),

    # 3
    tf.keras.layers.Dense(64, activation=activation, kernel_regularizer=regularizers.l2(0.002)),
    tf.keras.layers.Dropout(0.2),

    # 3
    tf.keras.layers.Dense(544, activation=activation, kernel_regularizer=regularizers.l2(0.0005)),
    tf.keras.layers.Dropout(0.1),

    # 3
    tf.keras.layers.Dense(832, activation=activation, kernel_regularizer=regularizers.l2(0.002)),
    tf.keras.layers.Dropout(0.5),


    tf.keras.layers.Dense(1)
])


def get_metrics():
    return ['mse', 'mae']


# ==== Retrain the Model

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


def compile_and_fit(model, loss, optimizer):
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=get_metrics()
    )
    cb = tf.keras.callbacks.EarlyStopping(
        monitor='mse',
        patience=20,
    )

    history = model.fit(
        final_dataset,
        epochs=epochs,
        callbacks=[cb]
    )
    return model, history


batches = [32]
losses = [
    tf.keras.losses.Huber(delta=1.0),
    # tf.keras.losses.mse,
]
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.0001,
    decay_steps=STEPS_PER_EPOCH * 1000,
    decay_rate=1,
    staircase=False)

optimizers = [
    tf.keras.optimizers.SGD(lr=0.008, momentum=0.9),
]

observe_file = "./observation_loss_optimisers.csv"
of = open(observe_file, mode="w")
observe_writer = csv.writer(of, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
observe_writer.writerow(['loss', 'optimizer', 'metric'])

for l1, loss in enumerate(losses):
    for o1, optimizer in enumerate(optimizers):
        for i in range(2):
            model, history = compile_and_fit(model, loss, optimizer)

            final_csv = f"./loss_opt/L-{l1}_O-{o1}_sub-{i}.csv"
            csv_file = open(final_csv, 'w')
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Employee_ID', TARGET])

            observe_writer.writerow([l1, o1, 0])

            for index, row in enumerate(test_data):
                pd = row[np.newaxis]
                prediction = model.predict(pd)
                results = [test_index[index], prediction[0][0]]
                writer.writerow(results)

            csv_file.close()

of.close()

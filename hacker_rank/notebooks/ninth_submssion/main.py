import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU
from sklearn.decomposition import PCA
from core.utils import compile_and_fit, reduce_dimensions, write_to_csv, prepare_dataset, get_forecast
from all_models import build_model
# ========================================================0=============================================================

train_path = '~/Downloads/hacker_rank/Dataset/Train.csv'
raw_train_df = pd.read_csv(train_path)

test_path = '~/Downloads/hacker_rank/Dataset/Test.csv'
raw_test_df = pd.read_csv(test_path)

index_column = 'Employee_ID'
train_index = raw_train_df.pop(index_column)
test_index = raw_test_df.pop(index_column)

# Merge DataFrame
df = pd.concat([raw_train_df, raw_test_df], ignore_index=True)

# Categorical column
categorical_columns = ['Gender', 'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess',
                       'Compensation_and_Benefits']
df = pd.get_dummies(df, columns=categorical_columns)

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

# ========================================================1=============================================================

split_index = 7000
original_df = df[:7000]
test_df = df.drop(original_df.index)

# Remove target column
TARGET = 'Attrition_rate'
train_labels = original_df.pop(TARGET).values
test_df.pop(TARGET)


# Scale
final_df = original_df.copy()
stats = final_df.describe().transpose()
final_df = (final_df - stats['mean']) / stats['std']
test_df = (test_df - stats['mean']) / stats['std']

# Convert to numpy
raw_final_data = final_df.to_numpy()
raw_test_data = test_df.to_numpy()

# ========================================================2=============================================================


class Configuration:
    def __init__(self):
        self.shuffle_buffer = 1000
        self.pca_components = 16
        self.batch_size = 32
        self.epochs = 1000

        self.activation = tf.keras.activations.sigmoid
        self.loss = tf.keras.losses.Huber(delta=1.0)
        self.lr_rate = 0.005
        self.optimizer = tf.keras.optimizers.SGD(lr=self.lr_rate, momentum=0.9)
        self.metrics = ['mse']


conf = Configuration()

train_data, test_data = reduce_dimensions(raw_final_data, raw_test_data, conf)
train_dataset = prepare_dataset(train_data, train_labels, conf)

for x, y in train_dataset:
    print(f"x: {x}, y: {y}")
    break

model = build_model(conf)

for i in range(2):
    model, history = compile_and_fit(model, conf, train_dataset)

    filename = f"./submissions/submission_{i}.csv"
    forecast = get_forecast(model, test_data)
    write_to_csv(test_index, forecast, filename)


# ========================================================2=============================================================
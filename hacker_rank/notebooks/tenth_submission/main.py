import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU
from sklearn.decomposition import PCA
from core.utils import compile_and_fit, reduce_dimensions, write_to_csv, prepare_dataset, get_forecast, norm
from core.config import Configuration
from all_models import build_model
# ========================================================0=============================================================

train_path = '~/Downloads/hacker_rank/Dataset/Train.csv'
raw_train_df = pd.read_csv(train_path)

test_path = '~/Downloads/hacker_rank/Dataset/Test.csv'
raw_test_df = pd.read_csv(test_path)

index_column = 'Employee_ID'
train_index = raw_train_df.pop(index_column)
test_index = raw_test_df.pop(index_column)
# ========================================================1=============================================================

mice_train_path = '/home/shravan/youtube_tutorials/tmp/hacker_earth/imputed_data/train_mice_imputed.csv'
mice_test_path = '/home/shravan/youtube_tutorials/tmp/hacker_earth/imputed_data/test_mice_imputed.csv'

knn_train_path = '/home/shravan/youtube_tutorials/tmp/hacker_earth/imputed_data/train_knn_imputed.csv'
knn_test_path = '/home/shravan/youtube_tutorials/tmp/hacker_earth/imputed_data/test_knn_imputed.csv'

mean_train_path = '/home/shravan/youtube_tutorials/tmp/hacker_earth/imputed_data/train_mean_imputed.csv'
mean_test_path = '/home/shravan/youtube_tutorials/tmp/hacker_earth/imputed_data/test_mice_imputed.csv'


# Raw Test DataFrames
raw_mice_test_df = pd.read_csv(mice_test_path)
raw_knn_test_df = pd.read_csv(knn_test_path)
raw_mean_test_df = pd.read_csv(mean_test_path)

# Raw Train DataFrames
raw_mice_train_df = pd.read_csv(mice_train_path)
raw_knn_train_df = pd.read_csv(knn_train_path)
raw_mean_train_df = pd.read_csv(mean_train_path)

# # Split Data
# mice_train_df = raw_mice_train_df.sample(frac=0.85, random_state=2)
# mice_valid_df = raw_mice_train_df.drop(mice_train_df.index)
#
#
# knn_train_df = raw_knn_train_df.sample(frac=0.85, random_state=2)
# knn_valid_df = raw_knn_train_df.drop(knn_train_df.index)
#
#
# mean_train_df = raw_mean_train_df.sample(frac=0.85, random_state=2)
# mean_valid_df = raw_mean_train_df.drop(mean_train_df.index)

mice_train_df = raw_mice_train_df.copy()
knn_train_df = raw_knn_train_df.copy()
mean_train_df = raw_mean_train_df.copy()

# Separate target
TARGET = 'Attrition_rate'
mice_train_labels = mice_train_df.pop(TARGET).values
# mice_valid_labels = mice_valid_df.pop(TARGET).values

knn_train_labels = knn_train_df.pop(TARGET).values
# knn_valid_labels = knn_valid_df.pop(TARGET).values

mean_train_labels = mean_train_df.pop(TARGET).values
# mean_valid_labels = mean_valid_df.pop(TARGET).values

# Scale Data
# mice_train_df, mice_valid_df, mice_test_df = norm(mice_train_df, mice_valid_df, raw_mice_test_df)
mice_train_df, mice_valid_df, mice_test_df = norm(mice_train_df, None, raw_mice_test_df)

# mice_valid_df = norm(mice_valid_df)

# knn_train_df, knn_valid_df, knn_test_df = norm(knn_train_df, knn_valid_df, raw_knn_test_df)
knn_train_df, knn_valid_df, knn_test_df = norm(knn_train_df, None, raw_knn_test_df)
# knn_valid_df = norm(knn_valid_df)

# mean_train_df, mean_valid_df, mean_test_df = norm(mean_train_df, mean_valid_df, raw_mean_test_df)
mean_train_df, mean_valid_df, mean_test_df = norm(mean_train_df, None, raw_mean_test_df)
# mean_valid_df = norm(mean_valid_df)

# Converting to numpy
mice_train_data = mice_train_df.to_numpy()
# mice_valid_data = mice_valid_df.to_numpy()
mice_test_data = mice_test_df.to_numpy()

knn_train_data = knn_train_df.to_numpy()
# knn_valid_data = knn_valid_df.to_numpy()
knn_test_data = knn_test_df.to_numpy()

mean_train_data = mean_train_df.to_numpy()
# mean_valid_data = mean_valid_df.to_numpy()
mean_test_data = mean_test_df.to_numpy()

conf = Configuration()

mice_train_dataset = prepare_dataset(mice_train_data, mice_train_labels, conf)
# mice_valid_dataset = prepare_dataset(mice_valid_data, mice_valid_labels, conf)

knn_train_dataset = prepare_dataset(knn_train_data, knn_train_labels, conf)
# knn_valid_dataset = prepare_dataset(knn_valid_data, knn_valid_labels, conf)

mean_train_dataset = prepare_dataset(mean_train_data, mean_train_labels, conf)
# mean_valid_dataset = prepare_dataset(mean_valid_data, mice_valid_labels, conf)

# datasets = [
#     [mice_train_dataset, mice_valid_dataset, mice_test_data, 'mice_model'],
#     [knn_train_dataset, knn_valid_dataset, knn_test_data, 'knn_model'],
#     [mean_train_dataset, mean_valid_dataset, mean_test_data, 'mean_model']
# ]

datasets = [
    [mice_train_dataset, None, mice_test_data, 'mice_model'],
    [knn_train_dataset, None, knn_test_data, 'knn_model'],
    [mean_train_dataset, None, mean_test_data, 'mean_model']
]

# ========================================================2=============================================================

model_architecture = build_model(conf)

for index in range(2):
    for i, data in enumerate(datasets):

        # Model Save Path
        save_path = f"./saved_models/{data[3]}"

        # Compile Model, Fit Model and Save the Model for forecasting
        compile_and_fit(model_architecture, conf, data[0], data[1], save_path)

        # Load the Saved Model and Predict
        forecast = get_forecast(save_path, data[2])

        # Write the prediction into CSV file
        filename = f"./submissions/submission_{i}_{index}.csv"
        write_to_csv(test_index, forecast, filename)


# for i in range(2):
#     model, history = compile_and_fit(model, conf, train_dataset, valid_dataset, save_path)
#
#     filename = f"./submissions/submission_{i}.csv"
#     forecast = get_forecast(save_path, test_data)
#     write_to_csv(test_index, forecast, filename)


# ========================================================2=============================================================
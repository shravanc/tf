from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime


from tensorflow import keras
import os
import re




def load_directory_data(directory):
  print("directory----->", directory)
  data = {}
  data["sentence"] = []
  #data["sentiment"] = []
  for file_path in os.listdir(directory):
    with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
      print("file_path--->", file_path)
      data["sentence"].append(f.read())
      #data["sentiment"].append(re.match("\d+.txt", file_path).group(1))

  print(data)
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
  pos_df = load_directory_data(os.path.join(directory, "pos"))
  neg_df = load_directory_data(os.path.join(directory, "neg"))
  print("-------------")
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


def download_and_load_datasets(force_download=False):
  dataset = tf.keras.utils.get_file(
      fname="kannada_dataset.tar.gz", 
      origin="https://github.com/shravanc/tf/blob/master/kannada_dataset_v1.tar.gz?raw=true", 
      extract=True)
 

  print("dataset--->", os.path.dirname(dataset)) 
  train_df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                       "kannada_dataset"))
  #test_df = load_dataset(os.path.join(os.path.dirname(dataset), 
  #                                    "aclImdb", "test"))
  
  return train_df


train = download_and_load_datasets()
print(train.head())

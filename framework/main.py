import numpy as np
import pandas as pd
import os
import time


import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

from lib.scaler_list import get_distributions
from lib.plotting import make_plot
from lib.cluster_methods import save_cluster
from lib.model_list import get_models
from lib.model import train_datasets

# Filename
filename = '/home/shravan/tf/datasets/parkinson_sample.csv'
df = pd.read_csv(filename)
all_columns = list(df.columns)

"""
Define csv_path and generate csv after performing 
each scaling on the datasets
"""
csv_path = "/home/shravan/tf/tf/framework/csv_files/"
def generate_csv_with_different_scaling_method(df, columns=[], generate_file=False):
  
  # gets all scaling method defined in this framework
  distributions = get_distributions(df)

  if generate_file:
    for title, X in distributions:
      print("title: ", title)
      df = pd.DataFrame(X, columns=all_columns)
      csv_filename = os.path.join(csv_path, title+'.csv')

      # This creates a CSV for each scaling
      df.to_csv(csv_filename)

  return distributions


image_path = "/home/shravan/tf/tf/framework/images"
all_csv = os.listdir(csv_path)
"""
Analysing Each scaling method. This will generate an image for two 
columns at a time. Columns that need to be examined can be changed
"""
def analyse_distribution():
  y = df['total_UPDRS']
  columns_to_analyse = [19, 20]
  distributions = get_distributions( df.to_numpy()[:, columns_to_analyse])
  for i, dist in enumerate(distributions):
    make_plot(i, distributions, y, image_path)

analyse_distribution()


"""
Analyse clusters for each scaling method
"""
cluster_path = "/home/shravan/tf/tf/framework/clusters"
def cluster_analysis():
  k = 3
  distributions = get_distributions(df)
  for i, dist in enumerate(distributions):
    save_cluster(k, dist[1], cluster_path, dist[0])

cluster_analysis()


def format(csv, model, score, time):
  csv_base_len      = 30
  model_base_length = 15

  scale = csv.split('.csv')[0].ljust(csv_base_len)
  model = model.ljust(model_base_length)
  score = round(score, 3)  * 100
  score = str(score)
  time  = str(time)

  data = f"{scale}, {model}, {score}, {time}\n"
  
  return data
  

"""
This method does all the hard word of training the each
model with each dataset generated for each scaling method.
This will then record all the information in an CSV for
firther analysis on the best combination.
"""
def train():
  f = open("train_data.csv", 'w')

  for csv in all_csv:
    abs_csv = os.path.join(csv_path, csv)
    df = pd.read_csv(abs_csv)
    models = get_models()
    for model in models:
      start = time.time()

      score = train_datasets(df, model)

      end = time.time()
      temp = end-start
      minutes = temp//60


      data = format(csv, model[0], score, minutes)
      f.write(data)
    

train()

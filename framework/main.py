import numpy as np
import pandas as pd
import os

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm

from lib.scaler_list import get_distributions
from lib.plotting import make_plot
from lib.cluster_methods import save_cluster
from lib.model import train_datasets, train_lasso, lasso_cv

filename = '/home/shravan/tf/datasets/parkinson_sample.csv'
df = pd.read_csv(filename)
all_columns = list(df.columns)

csv_path = "/home/shravan/tf/tf/framework/csv_files/"
def generate_csv_with_different_scaling_method(df, columns=[], generate_file=False):
  
  distributions = get_distributions(df)

  if generate_file:
    for title, X in distributions:
      df = pd.DataFrame(X, columns=all_columns)
      csv_filename = os.path.join(csv_path, title+'.csv')
      df.to_csv(csv_filename)

  return distributions
generate_csv_with_different_scaling_method(df, generate_file=True)



image_path = "/home/shravan/tf/tf/framework/images"
all_csv = os.listdir(csv_path)
print(all_csv)

y = df['total_UPDRS']
#df = df[:, [0, 5]]

def analyse_distribution():
  image_columns = [19, 20]
  distributions = get_distributions( df.to_numpy()[:, image_columns])
  for i, dist in enumerate(distributions):
    make_plot(i, distributions, y, image_path)

#analyse_distribution()


cluster_path = "/home/shravan/tf/tf/framework/clusters"
def cluster_analysis():
  k = 3
  distributions = get_distributions(df)
  for i, dist in enumerate(distributions):
    save_cluster(k, dist[1], cluster_path, dist[0])

#cluster_analysis()


def train_data():
  f = open("results.txt", "w")
  for csv in all_csv:
    print(csv)
    abs_csv = os.path.join(csv_path, csv)
    print(abs_csv)
    df = pd.read_csv(abs_csv)
    score = train_datasets(df) 
    f.write(f"Best CV score for {csv}: {score}")

  

#train_data()

d = "/home/shravan/tf/tf/framework/csv_files/nscaled_data.csv"
df = pd.read_csv(d)
#train_datasets(df)
#train_lasso(df)
lasso_cv(df)



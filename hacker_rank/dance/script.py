import os
import csv
import pandas as pd
import shutil


images = '/home/shravan/Downloads/dance_dataset/dataset/'

train_images = '/home/shravan/Downloads/dance_dataset/dataset/train.csv'
test_images = '/home/shravan/Downloads/dance_dataset/dataset/test.csv'


df = pd.read_csv(train_images)
print(df.head())

classes = df['target'].unique() #.values
print(classes)

base_dir = '/home/shravan/Downloads/dance_dataset/workspace'
# for cl in classes:
#     dir = os.path.join(base_dir, cl)
#     os.mkdir(dir)

# next(df.iterrows())
# for index, row in df.iterrows():
#     fname = row.values[0]
#     dance = row.values[1]
#     source = os.path.join(images, 'train', fname)
#     destination = os.path.join(base_dir, 'train', dance, fname)
#
#     print(source, destination)
#     shutil.move(source, destination)
    # break


print(len(df))
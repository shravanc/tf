import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Train Data
url = '/home/shravan/Downloads/hacker_rank/Dataset/hr_train.csv'
train_df = pd.read_csv(url)
train_index = train_df.pop('Employee_ID')

# Load Test Data
url = '/home/shravan/Downloads/hacker_rank/Dataset/hr_test.csv'
test_df = pd.read_csv(url)
test_index = test_df.pop('Employee_ID')

df = pd.concat([train_df, test_df], ignore_index=True)


split_index = 7
train_df = df[:7]
test_df = df[7:]

print(test_df.head())
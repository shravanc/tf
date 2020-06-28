import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import seaborn as sns
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Load Train Data
url = '/home/shravan/Downloads/hacker_rank/Dataset/Train.csv'
raw_train_df = pd.read_csv(url)
train_index = raw_train_df.pop('Employee_ID')

# Load Test Data
url = '/home/shravan/Downloads/hacker_rank/Dataset/Test.csv'
raw_test_df = pd.read_csv(url)
test_index = raw_test_df.pop('Employee_ID').values

df = pd.concat([raw_train_df, raw_test_df], ignore_index=True)

# Categorical_columns
categorical_columns = ['Gender', 'Relationship_Status', 'Hometown', 'Unit', 'Decision_skill_possess',
                       'Compensation_and_Benefits']
df = pd.get_dummies(df, columns=categorical_columns)
# print(df.head(2))
# sns.heatmap(df[['Age', 'Time_of_service', 'Time_since_promotion', 'growth_rate', 'Travel_Rate', 'Post_Level',
#                 'Pay_Scale']].corr())
# sns.heatmap(df.corr())
# plt.show()


# ===============Imputation================
# Imputation for Age
# Age is related to Time_of_service and Time_since_promotion
ptable = df.pivot_table(
    values='Age',
    index='Time_since_promotion',
    aggfunc=np.mean
)
print(ptable)
print(ptable.iloc[0].values[0])


def get_element(x):
    return ptable.iloc[int(x['Time_since_promotion'])].values[0]


df['Age'].fillna(
    df[df['Age'].isnull()].apply(
        get_element,
        axis=1
    ),
    inplace=True
)


# Imputation for Time_of_service
# Time_of_service is related to Age and Time_of_service
# print(df.isna().sum())
# sns.heatmap(df[['Age', 'Time_of_service', 'Time_since_promotion', 'growth_rate', 'Travel_Rate', 'Post_Level',
#                 'Pay_Scale']].corr())
# plt.show()

ptable = df.pivot_table(
    values='Time_of_service',
    index='Time_since_promotion',
    columns='Age',
    aggfunc=np.mean
)

print(ptable)
print(ptable.loc[2, 21])


def get_element(x):
    return ptable.loc[x['Time_since_promotion'], x['Age']]


df['Time_of_service'].fillna(
    df[df['Time_of_service'].isnull()].apply(
        get_element,
        axis=1),
    inplace=True
)


# For numeric column which are not related ro other fields
missing_numeric_columns = ['Pay_Scale', 'VAR2', 'VAR4', 'Work_Life_balance']
for col in missing_numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)


# =================End of Imputation===============================
import pandas as pd


url = "./clean_train.csv"
df = pd.read_csv(url)
print(df.head())

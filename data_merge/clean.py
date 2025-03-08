import pandas as pd

df = pd.read_csv('data.csv')

missing_percent = df.isnull().sum()/len(df) * 100
print(missing_percent)

df.dropna()

columns_to_drop = missing_percent[missing_percent > 30].index
print(columns_to_drop)

df = df.drop(columns = columns_to_drop)
print(df)

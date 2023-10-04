import pandas as pd
df = pd.read_csv("/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/database/database_k.csv")
print(df.head())
print(df.info())
print(df.isna().sum())
print(df.columns)
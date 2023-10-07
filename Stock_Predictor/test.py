import pandas as pd
df = pd.read_csv("/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/database/TOTAL_DATABASE.csv")
df.drop(columns={"Unnamed: 0"},inplace=True)
filepath = "/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/database/TOTAL_DATABASE_CLEAN.csv"
df.to_csv(path_or_buf=filepath)
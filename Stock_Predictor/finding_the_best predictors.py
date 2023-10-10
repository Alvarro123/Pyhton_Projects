import pandas as pd
import numpy as np
import json
filepath_1 = "/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/wig20_performance_data.json"
filepath_2 = "/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/currencies_performance_data.json"
df_1 = pd.read_json(filepath_1)
df_2 = pd.read_json(filepath_2)
df_1_heads = [i for i in df_1.columns if "Coefficients" in i]
df_2_heads = [i for i in df_2.columns if "Coefficients" in i]
coefficients = []
for col in df_1_heads:
    for i in range(0,5):
        value = df_1[col].iloc[i]
        coefficients.append(value[0])
for col in df_2_heads:
    for i in range(0,5):
        value = df_2[col].iloc[i]
        coefficients.append(value[0])
unique = []
for coef in coefficients:
    if coef in unique:
        continue
    else:
        unique.append(coef)
counts = []
for uniq in unique:
    counts.append(coefficients.count(uniq))
df = []
for i in range(len(unique)):
    df.append([])
    df[i].append(unique[i])
    df[i].append(counts[i])
data = pd.DataFrame(df, columns=["Symbols", "Counts"])
data = data.sort_values(by = "Counts", ascending=False).reset_index()
data = data.iloc[:5]
data = data["Symbols"]
path = "/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/best_coeff.csv"
data = data.to_csv(path_or_buf=path, mode = "x")
print(data)




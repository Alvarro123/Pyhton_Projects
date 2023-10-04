import os
import pandas as pd
path = "/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/database"
list_of_filenames = []
for (root, dirs, file) in os.walk(path):
    for f in file:
        if ".csv" in f:
            list_of_filenames.append(f)
#path generator
path_list = []
for file in list_of_filenames:
    file = str(file)
    string = "/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/database/{filename}".format(filename =file)
    path_list.append(string)
#merging the datasets
keys = list_of_filenames
values = []
for path in path_list:
    df =pd.read_csv(path)
    values.append(df["Date_index"].count())
the_highest = keys[values.index(max(values))]
index = list_of_filenames.index(str(the_highest))
df = pd.read_csv(path_list[3])["Date_index"]
df.drop(columns={"Unnamed: 0", "Data"}, inplace=True)
for i in range(len(path_list)):
    dt = pd.read_csv(path_list[i])
    dt.drop(columns={"Unnamed: 0", "Data"}, inplace=True)
    df = pd.merge(left = df, right = dt, left_on= "Date_index", right_on = "Date_index", how = "left")
#filling nan
keys = []
values = []
for col in df.columns:
    col = str(col)
    keys.append(col)
    values.append(0)
dict_nan = {k:v for k, v in zip(keys, values)}
df.fillna(dict_nan, inplace=True)
print(len(df.columns))
print(df.isna().sum())
print(df.info())
filepath = "/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/database/TOTAL_DATABASE.csv"
df.to_csv(path_or_buf=filepath, mode = "x")









import pandas as pd
symbols_and_names = pd.read_excel("symbols_names_test.xlsx")
symb = symbols_and_names["Symbols"].to_list()
nm = symbols_and_names["Names"].to_list()
symbols = []
for i in range(len(symb)):
    symbols.append(symb[i].lower())
historic_data_links = []
for i in range(len(symbols)):
    symbol = symbols[i]
    string = "https://stooq.pl/q/d/l/?s={s}&i=d".format(s = symbol)
    historic_data_links.append(string)
limiter = len(historic_data_links)
list_of_dataframes = []
for i in range(len(historic_data_links)):
    df = pd.read_csv(historic_data_links[i])
    df = df.loc[["Data", "Zamknięcie"]]
    df = df.rename(columns = {"Zamknięcie" : symb[i]})
    list_of_dataframes.append(df)

while limiter >0:
    for i in range(len(historic_data_links)):
        data = pd.read_csv(historic_data_links[i])
        values = data[["Data", "Zamkniecie"]]
        names = nm
        title = str(names[i] + " price")
        values = values.rename(columns={"Zamkniecie":title})
        list_of_dataframes.append(values)
        limiter = limiter - 1
df = list_of_dataframes[0]["Data"]
for i in list_of_dataframes:
    df = pd.merge(left = df, right = i, left_on = "Data", right_on = "Data", how = "left")
dates = df["Data"].tolist()
date_converted = df["Data"].str.split(pat = "-", expand = True, regex = None)
#wszystko trzeba przeliczyć na dni
date_converted.columns = ["Years","Months","Days"]
date_index = []
years = date_converted["Years"].tolist()
months = date_converted["Months"].tolist()
days = date_converted["Days"].tolist()
for i in range(len(days)):
    y_l = float(years[i]) * 365
    m_l = float(months[i]) * 30
    d_l = float(days[i])
    index = y_l + m_l + d_l
    date_index.append(index)
date_dic = {"Data":dates, "Date_index":date_index}
df_time = pd.DataFrame(date_dic)
#merging with dataset
df = pd.merge(left = df, right = df_time, left_on = "Data", right_on = "Data", how = "left")
#cleaning the dataset
for col in df.columns:
    condition = (df[col].isna() == False)
    df[col] = df[col][condition]
    if df[col].dtype != "float64":
        df.drop(columns = col, inplace=True)
keys = []
values = []
for col in df.columns:
    keys.append(str(col))
    values.append(0)
nan_dic = {k:v for k,v in zip(keys,values)}
df.fillna(nan_dic,inplace=True)
database = df
filepath = "/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/database.csv"
database.to_csv(path_or_buf=filepath, mode = "x")






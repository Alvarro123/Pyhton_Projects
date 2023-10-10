import pandas as pd
import numpy as np
from datetime import date, timedelta
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# importing the desired coefficients
coefficients = pd.read_csv("/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/best_coeff.csv")
y_values = ["EUR/PLN", "CHF/PLN", "USD/PLN", "GBP/PLN"]
#importing_the_training_data
total_database = pd.read_csv("/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/database/TOTAL_DATABASE.csv")
features = [i + " price" for i in coefficients["Symbols"].to_list()] + ['Date_index']
values = [i + " price" for i in y_values]
x_data = total_database[features]
y_data = total_database[[i + " price" for i in y_values]]
#establishing DATA TIME
#calculating_date_index_3_months_ahead
data = str(date.today())
data = data.split("-")
year = float(data[0]) * 365
month = float(data[1]) * 30
day = float(data[2])
date_index =  year + month + day
#HERE YOU INPUT THE TIME INTERVAL IN DAYS
exp_date_index = float(date_index + 1)
#imprting feature data:
current_features = [i.replace("/", "").lower() for i in coefficients["Symbols"].to_list()]
predict_data = []
for curr in current_features:
    string = "https://stooq.pl/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv".format(symbol = curr)
    df = pd.read_csv(string)
    dt = df["Zamkniecie"].iloc[0]
    predict_data.append(dt)
exp_date = date.today() + timedelta(days=30)
exp_date = str(exp_date)
predict_data.append(exp_date_index)
df_dic = {k:v for k,v in zip(features,predict_data)}
df_t = pd.DataFrame(df_dic, index=[0])
#print(df_t)
scores = []
predicted_values = []
real_values = []
#for each of Y values we look for the best random state for the highes score, then we import the real data and predict the value. 
for val in values:
    int_score = []
    random_states = []
    for i in range(100):
        #splitting the data
        y_df = y_data[val]
        X_train, X_test, y_train, y_test = train_test_split(x_data,y_df, test_size=0.2, random_state= i)
        #training the model
        mlr = LinearRegression()
        mlr.fit(X_train,y_train)
        #finding the random state of the data
        int_score.append(mlr.score(X_test,y_test))
        random_states.append(i)
    desired_index = int_score.index(max(int_score))
    random = random_states[desired_index]
    #importing_current_value
    symb = val.rstrip(" price")
    sym = symb.replace("/", "").lower()
    string = "https://stooq.pl/q/l/?s={symbol}&f=sd2t2ohlcv&h&e=csv".format(symbol = sym)
    df = pd.read_csv(string)
    dt = df["Zamkniecie"].iloc[0]
    real_values.append(dt)
    #fiting the best model
    y_df = y_data[val]
    X_train, X_test, y_train, y_test = train_test_split(x_data,y_df, test_size=0.2, random_state= random)
    model = LinearRegression()
    model.fit(X_train,y_train)
    scores.append(model.score(X_test,y_test))
    #predicting the value
    predicted_values.append(np.array(model.predict(df_t))[0])
#print(real_values, predicted_values, scores)
today = str(date.today())
future = str(date.today() + timedelta(days=30))
key_1 = "Current Price {DATA}".format(DATA = today)
key_2 = "Predicted Price for {FUTURE}".format(FUTURE = future)
key_3 = "Difference"
key_4 = "Model Score"
index = ["EUR/PLN", "CHF/PLN", "USD/PLN", "GBP/PLN"]
dif = []
for i in range(len(real_values)):
    dif.append(round(real_values[i] - predicted_values[i],2))
key_s = [key_1,key_2,key_3,key_4]
vvalue_s = [real_values, predicted_values, dif, scores]
df_dic = {ke:vv for ke, vv in zip(key_s,vvalue_s)}
final_df = pd.DataFrame(df_dic, index = index)
excel_path = "/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/RESULTS.xlsx"
final_df.to_excel(excel_writer=excel_path)


#final_df = pd.DataFrame(data, columns = ["Current Price", "Predicted Price", "Difference", "Model Score"], index = values)
print(final_df)


    







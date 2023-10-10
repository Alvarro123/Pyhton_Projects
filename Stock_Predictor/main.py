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
#predicting feature data
s_x = []
predict_data = []
internal_features = [i + " price" for i in coefficients["Symbols"].to_list()]
for x_val in internal_features:
    int_sc = []
    rand_om_states = []
    X = np.array(total_database['Date_index']).reshape(-1,1)
    y = np.array(total_database[x_val]).reshape(-1,1)
    for f in range(100):
        #splitting the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=f)
        #training the model
        mlr = LinearRegression()
        mlr.fit(X_train,y_train)
        #finding the random state of the data
        int_sc.append(mlr.score(X_test,y_test))
        rand_om_states.append(f)
    desi_red_index = int_sc.index(max(int_sc))
    rand_om = rand_om_states[desi_red_index]
    #creating_dataframe_to_predict
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state= rand_om)
    model = LinearRegression()
    model.fit(X_train,y_train)
    s_x.append(model.score(X_test,y_test))
    #predicting the value
    predict_data.append(np.array(model.predict(np.array(exp_date_index).reshape(-1,1)))[0])

exp_date = date.today() + timedelta(days=360)
exp_date = str(exp_date)
predict_data.append(exp_date_index)
df_dic = {k:v for k,v in zip(features,predict_data)}
df_t = pd.DataFrame(df_dic, index=[0])
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
weightened_scores = [sc*statistics.median(s_x) for sc in scores]
data = []
for v in values:
    data.append([])
difference = []
for i in range(len(predicted_values)):
    difference.append(round((real_values[i]-predicted_values[i]),2))
for r in range(len(data)):
    data[r].append(real_values[i])
    data[r].append(predicted_values[i])
    data[r].append(difference[i])
    data[r].append(weightened_scores[i])



final_df = pd.DataFrame(data, columns = ["Current Price", "Predicted Price", "Difference", "Model Score"], index = values)
print(final_df)


    







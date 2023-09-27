import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("manhattan.csv")
#print(df.head())
df = pd.DataFrame(df)
from sklearn.model_selection import train_test_split
x = pd.DataFrame(df[["bedrooms","bathrooms","size_sqft","min_to_subway", "floor","building_age_yrs","no_fee","has_roofdeck", "has_washer_dryer","has_doorman","has_elevator","has_dishwasher","has_patio","has_gym"]])
y = pd.DataFrame(df["rent"])
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=6)
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)
from sklearn.linear_model import LinearRegression
mlr = LinearRegression()
mlr.fit(x_train,y_train)
y_predict = mlr.predict(x_test)
sonny_apartment = [[1, 1, 620, 16, 1, 98, 1, 0, 1, 0, 0, 1, 1, 0]]
predict = mlr.predict(sonny_apartment)
#print("Predicted rent: $%.2f" % predict)
plt.scatter(y_test, y_predict, alpha = 0.4)
plt.xlabel("Prices: Yi")
plt.ylabel("Predicted prices: Y")
plt.title("Actual Rent vs Predicted Rent")
#plt.show()
print(mlr.coef_)
plt.scatter(df[["bathrooms"]],df[["rent"]],alpha = 0.4)
plt.scatter(df[["has_doorman"]],df[["rent"]],alpha = 0.4)
plt.scatter(df[["has_elevator"]],df[["rent"]],alpha = 0.4)
#plt.show()
print(mlr.score(x_train, y_train))
print(mlr.score(x_test, y_test))
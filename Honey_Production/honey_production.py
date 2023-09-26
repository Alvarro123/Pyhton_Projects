from sklearn import linear_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("US_honey_dataset_updated.csv")
#print(dataset.head())
#dataset.info()
prod_year = dataset.groupby("year").agg({"value_of_production":"mean"}).reset_index()
#print(prod_year.head())
X = prod_year["year"]
y = prod_year["value_of_production"]
plt.plot(X,y)
plt.xlabel("Years")
plt.ylabel("Honey Mean Volume")
plt.show()


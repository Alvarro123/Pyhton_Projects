from sklearn import linear_model
import pandas as pd
import numpy as np
dataset = pd.read_csv("US_honey_dataset_updated.csv")
print(dataset.head())
dataset.info()
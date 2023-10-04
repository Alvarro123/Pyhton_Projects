import os
path = "/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/database/missing_data"
list = []
for (root, dirs, file) in os.walk(path):
    for f in file:
        if ".csv" in f:
            list.append(f)
print(list)

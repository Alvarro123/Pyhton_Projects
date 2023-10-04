import os
path = "/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/database"
list_of_filenames = []
for (root, dirs, file) in os.walk(path):
    for f in file:
        if ".csv" in f:
            list_of_filenames.append(f)
print(list_of_filenames)


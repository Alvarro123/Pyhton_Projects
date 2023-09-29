
url_1 = "/Users/test/Desktop/Pyhton_Projects/KGHM_Linera_Predictor_Project"
class Stock:
    def get_list_of_files_from_stooq(self,url):
        self.url = url
        import os
        arr = os.listdir(self.url)
        import glob
        csvfiles = [f for f in glob.glob("*.csv")]
        lst_names = [n.rstrip("_d.csv") for n in csvfiles]
        Stock.list_of_values = lst_names
        Stock.csvfiles = csvfiles
        return lst_names

    def import_files_from_stooq(self):
        import pandas as pd
        limiter = len(self.csvfiles)
        list_of_dataframes = []
        while limiter >0:
            for i in range(len(self.csvfiles)):
                data = pd.read_csv(self.csvfiles[i])
                values = data[["Data", "Zamkniecie"]]
                names = self.list_of_values
                title = str(names[i] + " price")
                values = values.rename(columns={"Zamkniecie":title})
                list_of_dataframes.append(values)
                limiter = limiter - 1
        df = list_of_dataframes[0]["Data"]
        for i in list_of_dataframes:
            df = pd.merge(left = df, right = i, left_on = "Data", right_on = "Data", how = "left")
        Stock.raw_dataset_from_stooq = df
        return df
    
    def clean_stooq_dataset(self):
        import pandas as pd
        self.dataset = Stock.raw_dataset_from_stooq
        for col in self.dataset.columns:
            condition = (self.dataset[col].isna() == False)
            self.dataset[col] = self.dataset[col][condition]
            if self.dataset[col].dtype != "float64":
                self.dataset.drop(columns = col, inplace=True)
        keys = []
        values = []
        for col in self.dataset.columns:
            keys.append(str(col))
            values.append(0)
        nan_dic = {k:v for k,v in zip(keys,values)}
        self.dataset.fillna(nan_dic,inplace=True)
        Stock.clean_dataset = self.dataset
        return self.dataset
    def train_algorithm(self, y_value):
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        y_value = str(y_value)
        Y = self.clean_dataset[y_value]
        features = []
        for col in self.clean_dataset.columns:
            if col != y_value:
                features.append(col)
            else:
                continue
        X = self.clean_dataset[features]
        #Choosing the best random state
        #end_point of range could be as far as 4294967296. However due to  low computer power, it was limited to 100. 
        random_states = [i for i in range(0,100)]
        train_scores = []
        test_scores = []
        for i in random_states:
            X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.2 ,random_state = i)
            mlr = LinearRegression()
            mlr.fit(X_train,y_train)
            train_scores.append(mlr.score(X_train,y_train))
            test_scores.append(mlr.score(X_test,y_test))
        if max(train_scores) < max(test_scores):
            desired_index = test_scores.index(max(test_scores))
            desired_random = random_states[desired_index]
            
        else:
            desired_index = train_scores.index(max(train_scores))
            desired_random = random_states[desired_index]
        #Choosing better test split proportion
        test_sizes = [float(round(i,3)) for i in np.arange(0.005,0.9,0.005)]
        train_scor = []
        test_scor = []
        for i in test_sizes:
            X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = i,random_state = desired_random)
            mlr = LinearRegression()
            mlr.fit(X_train,y_train)
            train_scor.append(mlr.score(X_train,y_train))
            test_scor.append(mlr.score(X_test,y_test))
        if max(train_scor) < max(test_scor):
            desired_test_size_index = test_scor.index(max(test_scor))
            desired_test_size = test_sizes[desired_test_size_index]
        else:
            desired_test_size_index = train_scor.index(max(train_scor))
            desired_test_size = test_sizes[desired_test_size_index]
        test_size = desired_test_size
        X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = test_size, random_state=desired_random)
        mlr = LinearRegression()
        mlr.fit(X_train,y_train)
        train_score = mlr.score(X_train,y_train)
        test_score = mlr.score(X_test,y_test)
        coefficients = sorted(list(zip(features,mlr.coef_)),key = lambda x: abs(x[1]),reverse=True)
        return print("\n","Train Score: ",train_score,"\n", "Test Score: ", test_score,"\n","Coefficients: ", coefficients,"\n", "Test_size: ",test_size,"\n", "Random State: ", desired_random)
    
    

        
        




            
test = Stock()
test.get_list_of_files_from_stooq(url_1)
test.import_files_from_stooq()
test.clean_stooq_dataset()
test.train_algorithm("kgh price")



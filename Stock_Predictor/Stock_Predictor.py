class Stooq:
    def import_database(self,url):
        import pandas as pd
        Stooq.database = pd.read_csv(url)
        trash = "Unnamed: 0"
        if trash in Stooq.database.columns:
            Stooq.database.drop(columns={"Unnamed: 0"}, inplace=True)
        #print("\n","Database informations: ",Stooq.database.info(), "\n","Database Columns: ", Stooq.database.columns, "\n", "Number of Nan's", Stooq.database.isna().sum())
    def wig_20(self):
        # rename columns
        import pandas as pd
        symbols_names = pd.read_excel("symbols_names.xlsx")
        symbols = symbols_names["Symbols"]
        names = symbols_names["Names"]
        columns = [col.rstrip(" price") for col in Stooq.database.columns]
        Stooq.database.columns = columns
        #model development
        values = ["EUR/PLN", "CHF/PLN", "USD/PLN", "GBP/PLN"]
        features = []
        for col in Stooq.database.columns:
            if col in values:
                continue
            else:
                features.append(col)
        #temporarily only for 1 y
        import pandas as pd
        import numpy as np
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        #creating dataframe for merge start
        #df_dic = {"Company": values}
        #start_df = pd.DataFrame(data = df_dic)
        
        #creating_json_to_save_performance_results
        lst_of_results = []
        keys = []

        self.clean_dataset = Stooq.database
        for val in values:
            y_value = str(val)
            Y = self.clean_dataset[y_value]
            features = []
            for col in self.clean_dataset.columns:
                if col != y_value:
                    features.append(col)
                else:
                    continue
            X = self.clean_dataset[features]
            #Choosing the best random state
            # #end_point of range could be as far as 4294967296. However due to  low computer power, it was limited to 100. 
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
            the_most_revelant_coefficients = coefficients[0:5]
            #gen_df_dic = {"Company": y_value, "Best Predictor":the_most_revelant_coefficients[0][0],"Coefficient":the_most_revelant_coefficients[0][1]}
            
            #creating Json File to update the resutls
            key1 = y_value + " Coefficients"
            keys.append(key1)
            lst_of_results.append(coefficients)
            key2 = y_value + " Train Score"
            keys.append(key2)
            lst_of_results.append(train_score)
            key3 = y_value + " Test Score"
            keys.append(key3)
            lst_of_results.append(test_score)
            key4 = y_value + " Test Size"
            keys.append(key4)
            lst_of_results.append(test_size)
            key5 = y_value + " Random State"
            keys.append(key5)
            lst_of_results.append(desired_random)
            print("Y:", y_value,"\n","Train Score: ",train_score,"\n", "Test Score: ", test_score,"\n","Five the most relevant Coefficients: ",the_most_revelant_coefficients ,"\n", "Test_size: ",test_size,"\n", "Random State: ", desired_random)
        result_dic = {k:v for k, v in zip(keys, lst_of_results)}
        import json
        dict = result_dic
        with open("/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/currencies_performance_data.json","x") as outfile:
            json.dump(dict, outfile)
        




test = Stooq()
test.import_database("/Users/test/Desktop/Pyhton_Projects/Stock_Predictor/database/TOTAL_DATABASE.csv")
test.wig_20()

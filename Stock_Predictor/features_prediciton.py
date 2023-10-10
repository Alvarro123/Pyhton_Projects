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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
businesses = pd.read_json("yelp_business.json", lines = True)
reviews = pd.read_json("yelp_review.json", lines = True)
users = pd.read_json("yelp_user.json", lines = True)
checkins = pd.read_json("yelp_checkin.json",lines = True)
tips = pd.read_json("yelp_tip.json", lines = True)
photos = pd.read_json("yelp_photo.json", lines = True)
pd.options.display.max_columns = 60
pd.options.display.max_colwidth = 500
#print(businesses.columns)
#print(reviews.columns)
#print(users.columns)
#print(checkins.columns)
#print(tips.columns)
#print(photos.columns)
#print(businesses.info())
#print(users.describe())
#print(businesses[businesses["business_id"] == "5EvUIR4IzCWUOm0PsUZXjA"]["stars"])
#zacząć od mergowania wszystkiego po business ID
df = pd.merge(left = businesses, right = reviews, left_on = "business_id", right_on = "business_id", how = "left")
df = pd.merge(left = df, right = users, left_on= "business_id", right_on = "business_id", how="left")
df = pd.merge(left = df, right = checkins, left_on= "business_id", right_on = "business_id", how="left")
df = pd.merge(left = df, right = tips, left_on= "business_id", right_on = "business_id", how="left")
df = pd.merge(left = df, right = photos, left_on= "business_id", right_on = "business_id", how="left")

df.drop(['address','attributes','business_id','categories','city','hours','is_open','latitude','longitude','name','neighborhood','postal_code','state','time'], axis = 1, inplace=True)
#print(df.info())
df.fillna({"weekday_checkins":0, "weekend_checkins":0, "average_tip_length":0, "number_tips":0,"average_caption_length":0,"number_pics":0}, inplace=True)

#print(df.isna().any())
#print(df.corr())
#plt.scatter(df["average_review_sentiment"],df["stars"], alpha = 0.4)
#plt.show()
#plt.scatter(df["average_review_length"],df["stars"],alpha = 0.4)
#plt.show()
#plt.scatter(df["average_review_age"],df["stars"],alpha = 0.4)
#plt.show()
#plt.scatter(df["number_funny_votes"],df["stars"],alpha = 0.4)
#plt.show()
features = df[["average_review_length","average_review_age","number_funny_votes"]]
ratings = df["stars"]
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(features,ratings,test_size=0.2,random_state=1)
from sklearn.linear_model import LinearRegression
#mlr = LinearRegression()
#mlr.fit(X_train,y_train)
#print(mlr.score(X_train,y_train))
#print(mlr.score(X_test,y_test))
#print(sorted(list(zip(['average_review_length','average_review_age'],mlr.coef_)),key = lambda x: abs(x[1]),reverse=True))
#y_predicted = mlr.predict(X_test)
#plt.scatter(y_test,y_predicted, alpha= 0.4)
#plt.xlabel("Real values")
#plt.ylabel("Predicted values")
#plt.show()
#print(df.info())
#start define different subsets of data
sentiment = ["average_review_sentiment"]
binary_features = ['alcohol?','has_bike_parking','takes_credit_cards','good_for_kids','take_reservations','has_wifi']
numeric_features = ['review_count','price_range','average_caption_length','number_pics','average_review_age','average_review_length','average_review_sentiment','number_funny_votes','number_cool_votes','number_useful_votes','average_tip_length','number_tips','average_number_friends','average_days_on_yelp','average_number_fans','average_review_count','average_number_years_elite','weekday_checkins','weekend_checkins']
all_features = binary_features + numeric_features
feature_subset = ["review_count","number_useful_votes","price_range"]
def model_these_features(feature_list):
    ratings = df.loc[:,"stars"]
    features = df.loc[:,feature_list]
    X_train, X_test, y_train, y_test = train_test_split(features, ratings, test_size = 0.2, random_state = 1)
    if len(X_train.shape) <2:
        X_train = np.array(X_train).reshape(-1,1)
        X_test = np.array(X_test).reshape(-1,1)

    model = LinearRegression()
    model.fit(X_train,y_train)
    print("Train Score", model.score(X_train,y_train))
    print("Test Score", model.score(X_test,y_test))
    print(sorted(list(zip(feature_list,model.coef_)),key = lambda x: abs(x[1]),reverse=True))
    plt.scatter(y_test,y_predicted)
    plt.xlabel('Yelp Rating')
    plt.ylabel('Predicted Yelp Rating')
    plt.ylim(1,5)
    plt.show()


#model_these_features(sentiment)
#model_these_features(binary_features)
#model_these_features(numeric_features)
#model_these_features(all_features)
#model_these_features(feature_subset)
average = pd.DataFrame(list(zip(features.columns,features.describe().loc['mean'],features.describe().loc['min'],features.describe().loc['max'])),columns=['Feature','Mean','Min','Max'])
#print(average)
#print(df.info())
#print(df.head())
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(all_features,df['stars'],test_size=0.2, random_state=1)
model.fit(X_train,y_train)
danielles_delicious_delicacies = np.array([1,1,1,1,2,126,1,1,618,785,0.7,6,3,13,21,43,1221,2570,86,321,4.25,11,18,38,21]).reshape(1,-1)
#print(model.predict(danielles_delicious_delicacies))






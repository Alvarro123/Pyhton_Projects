import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
all_tweets = pd.read_json("random_tweets.json", lines=True)
#print(len(all_tweets))

#print(all_tweets.loc[0]["text"])
#print(all_tweets["retweet_count"].median())
all_tweets["is_viral"] = all_tweets["retweet_count"].apply(lambda tweet: 1 if tweet > 13 else 0)
#print(all_tweets["is_viral"].value_counts())
all_tweets["tweet_length"] = all_tweets["text"].apply(lambda tweet: len(tweet))
#print(all_tweets["tweet_length"].head())
lst_users = all_tweets["user"].to_list()
followers_count = []
for i in range(len(lst_users)):
    number = lst_users[i]["followers_count"]
    followers_count.append(number)
dt_dic_1 = {"followers_count": followers_count}
index = [i for i in range(len(followers_count))]
dt_pd_1 = pd.DataFrame(dt_dic_1, index=index)
all_tweets["followers_count"] = dt_pd_1["followers_count"]
friends_count = []
for i in range(len(lst_users)):
    number = lst_users[i]["friends_count"]
    friends_count.append(number)
dt_dic_2 = {"friends_count":friends_count}
dt_pd_2 = pd.DataFrame(dt_dic_2, index = index)
all_tweets["friends_count"] = dt_pd_2["friends_count"]
all_tweets["word_list"] = all_tweets["text"].apply(lambda tweet: tweet.split(" "))
all_tweets["word_count_in_tweet"] = all_tweets["word_list"].apply(lambda lst: len(lst))
# our features are: folowers_count, friends_count, worud_count_in_tweet
# now let's normalize the data
labels = all_tweets["is_viral"]
data = all_tweets[["followers_count", "friends_count", "word_count_in_tweet","tweet_length"]]
scaled_data = scale(data,axis=0)
train_data, test_data, train_labels, test_labels = train_test_split(scaled_data, labels, test_size=0.2, random_state=1)
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(train_data,train_labels)
score = classifier.score(test_data, test_labels)
scores = []
for k in range(1,201):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(train_data,train_labels)
    scores.append(classifier.score(test_data,test_labels))
desired_k = scores.index(max(scores))
print(desired_k)
x_values = [i for i in range(1,201)]
plt.xlabel("K value")
plt.ylabel("Score")
plt.scatter(x_values,scores, alpha = 0.4)
plt.show()

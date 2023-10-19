import pandas as pd
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
#ZACZĄĆ OD TYCH 3 CUSTOMOWYCH FEATEROW!!

print(all_tweets["friends_count"].head())
#print(lst_users[0]["followers_count"])
#all_tweets["followers_count"] = all_tweets["user"]["followers_count"].apply(lambda count: count)

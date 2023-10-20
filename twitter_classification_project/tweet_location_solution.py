import pandas as pd
new_york_tweets = pd.read_json("new_york.json", lines = True)
#print(len(new_york_tweets))
#print(new_york_tweets.columns)
#print(new_york_tweets.loc[12]["text"])
paris_tweets = pd.read_json("paris.json", lines = True)
#print(len(paris_tweets))
#print(paris_tweets.columns)
london_tweets = pd.read_json("london.json", lines = True)
#print(len(london_tweets))
#print(london_tweets.columns)
new_york_text = new_york_tweets["text"].to_list()
paris_text = paris_tweets["text"].to_list()
london_text = london_tweets['text'].to_list()
all_tweets = new_york_text + paris_text + london_text
labels = [0] * len(new_york_text) + [1] * len(paris_text) + [2] * len(london_text)
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(all_tweets,labels,test_size=0.2, random_state=1)
from sklearn.feature_extraction.text import CountVectorizer
counter = CountVectorizer()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)
#print(train_data[3], train_counts[3])
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(train_counts,train_labels)
predictions = classifier.predict(test_counts)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_labels,predictions)
#print(accuracy)
from sklearn.metrics import confusion_matrix
#print(confusion_matrix(test_labels,predictions))
tweet = "Postgraduate medical internship at Borowska Hospital has no sense and is waste of time!"
tweet_counts = counter.transform([tweet])
prediction = classifier.predict(tweet_counts)
print(prediction)

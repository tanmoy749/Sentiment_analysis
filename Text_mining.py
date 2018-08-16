import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

%matplotlib inline
train  = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')
train.head()

combi = train.append(test, ignore_index=True)
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern,input_txt)
    for i in r:
        input_txt = re.sub(i,'',input_txt)
    return input_txt

#remove twitter handles(@user)
combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#","")
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x:''.join([w for w in x.split() if len(w)>3]))
combi.head()

tokenized_tweet = combi['tidy_tweet'].apply(lambda x:x.split())
tokenized_tweet.head()

from nltk.stem.porter import *
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) #stemming
tokenized_tweet.head()

# Understanding the common words

all_words = ''.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# words in non racist/sexist tweets

normal_words=''.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

# racist/sexist tweets

negative_words = ''.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
wordcloud = WordCloud(width=800, height=500,random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud,interpolation="bilinear")
plt.axis('off')
plt.show()

# function to collect hashtags

def hashtag_extract(x):
    hashtags = []
    #loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+), i")
        hashtag.append(ht)
    return hashtags

#extracting hashtags from non racist/sexist tweets
HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])

#extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

#unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negtive,[])

# Non-Racist/Sexist tweets:
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# Selecting top 10 most frequent hashtags
d = d.nlargest(columns="Count", n=10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y= "Count")
ax.set(ylabel = 'Count')
plt.show()

#Racist/Sexist tweets:
b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()),
                  'Count': list(b.values())})
#Selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n=10)
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y= "Count")
ax.set(ylabel = 'Count')
plt.show()

# bag-of-word feature
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# bag-of-words feature matrix
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])

# TF-IDF features
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])


# Building model using Bag-of-words features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_bow = bow[:31962,:]
train_bow = bow[31962:,:]

#splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow,train['label'], random_state=42, test_size=0.3)
lreg = LogisticRegression()
lreg.fit(xtrain_bow,ytrain) #training the model

prediction = lreg.predict_proba(xvalid_bow) #predicting the validation set
prediction_int = prediction[:,1]>=0.3 #if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)

f1_score(yvalid, prediction_int) #calculating f1 score


# Building model using TF-IDF features


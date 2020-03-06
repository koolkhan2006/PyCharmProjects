#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import emot
import string
import pickle
import re
import nltk
nltk.download('wordnet')

train = pd.read_csv("train.csv", encoding = "Latin - 1")
print(train.head())

# rows and columns of train
print(train.shape)

# counts of sentiment in train dataset
print(train['sentiment'].value_counts())

#checking imbalance in the train dataset
print((train['sentiment'].value_counts()/train['sentiment'].shape)*100)

#notations
# 0 - Negative,1-Neutral,2-Positive,3 - Can't say

#check null values in train dataset
train.isnull().sum()

#removing null or blank data row from train data set
print("*"*50)
print("Check count for missing values in each column and drop that rows")
print("*"*50)
print(train.isnull().sum())
train = train.dropna()
print(train.isnull().sum())
train = train.reset_index(drop=True)
print(train.shape)

print(train.shape)

# lowercase of tweet column
train["tweet"] = train["tweet"].str.lower()
#checking datatype of tweet column
print(train.dtypes)
train['tweet'] = train['tweet'].astype(str)
#changing column width to get full display
pd.set_option('display.max_colwidth', -1)

# emoticons funtion
def convert_emoticons(text):
    for emoti in emot.EMOTICONS:
        text = re.sub(u'('+emoti+')', "_".join(emot.EMOTICONS[emoti].replace(",","").split()), text)
    return text

# remove emoticons
train['tweet'] = train['tweet'].apply(lambda x : convert_emoticons(str(x)))

# remove handles function
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt

train['tweet'] = train['tweet'].astype(str)
train['tweet'] = np.vectorize(remove_pattern)(train['tweet'], "@[\w]*") 
train['tweet'] = np.vectorize(remove_pattern)(train['tweet'], "#[\w]*")

#remove special characters and punctuations
def remove_punct(text):
    text_nopunct="".join([char for char in text if char not in string.punctuation])
    return text_nopunct

train['tweet'] = train['tweet'].astype(str)
train['tweet'] = train['tweet'].apply(lambda x: re.sub(r'[^\w\s]',' ',x))
train['tweet'] = train['tweet'].apply(lambda x: remove_punct(x))
train['tweet'] = train['tweet'].str.replace("[^a-zA-Z#]", " ")

#removing most important hashtags
train['tweet']=train['tweet'].str.lower()
train['tweet']=train['tweet'].str.replace("sxsw"," ")
train['tweet']=train['tweet'].str.replace("link"," ")
train['tweet']=train['tweet'].str.replace("quot"," ")
train['tweet']=train['tweet'].str.replace("amp"," ")
train['tweet']=train['tweet'].str.replace("google"," ")
train['tweet']=train['tweet'].str.replace("ipad"," ")
train['tweet']=train['tweet'].str.replace("iphone"," ")
train['tweet']=train['tweet'].str.replace("apple"," ")
train['tweet']=train['tweet'].str.replace("app"," ")
train['tweet']=train['tweet'].str.replace("called"," ")
train['tweet']=train['tweet'].str.replace("austin"," ")
train['tweet']=train['tweet'].str.replace("store"," ")
train['tweet']=train['tweet'].str.replace("android"," ")
train['tweet']=train['tweet'].str.replace("pop"," ")
train['tweet']=train['tweet'].str.replace("via"," ")
train['tweet']=train['tweet'].str.replace("social"," ")
train['tweet']=train['tweet'].str.replace("network"," ")
train['tweet']=train['tweet'].str.replace("mention"," ")
train['tweet']=train['tweet'].str.replace("new"," ")

#remove words with less than 2 characters
train['tweet'] = train['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>2]))

#tokenisation function

def tokenize(text):
    tokens=re.split('\W+',text)
    return tokens

train['tweet']=train['tweet'].apply(lambda x:tokenize(x.lower()))

# In[87]:
stopword=nltk.corpus.stopwords.words('english')

#remove stopwords
def remove_stopwords(tokenized_list):
    text=[word for word in tokenized_list if word not in stopword]
    return text
train['tweet']=train['tweet'].apply(lambda x:remove_stopwords(x))

#lemmatizer function
wn=nltk.WordNetLemmatizer()
def lemmatizing(tokenized_text):
    text=[wn.lemmatize(word) for word in tokenized_text]
    return text

train['tweet']=train['tweet'].apply(lambda x:lemmatizing(x))

#Convert list to str
train_string = pd.Series([' '.join(x) for x in train['tweet']])

train['tweet'] = train_string

# Word cloud for all words
train['tweet'] = train['tweet'].astype(str)

corpus = []
for i in range(0,len(train)):
    corpus.append(train['tweet'][i])

# tfidf = TfidfVectorizer(max_features=5250)
# X = tfidf.fit_transform(corpus).toarray()
# pickle.dump(tfidf, open('tfidf.pkl','wb'))
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(corpus).toarray()
pickle.dump(cv, open('cv.pkl','wb'))
y =  train['sentiment'].values
print(X.shape)
print(y.shape)

print("*"*50)
print("Dividing the dataset into train and test")
print("*"*50)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

print("*"*50)
print("Apply Naiive Bayes classification")
print("*"*50)
nb = MultinomialNB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)
print(nb.score(X,y))
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
print(cm)
print(cr)

# Saving model to disk
pickle.dump(nb, open('TwitterSentimentAnalysisFinalModel.pkl','wb'))
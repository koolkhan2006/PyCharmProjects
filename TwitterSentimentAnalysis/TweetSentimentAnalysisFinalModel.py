#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import emot
import string
import pickle
import re
import nltk
nltk.download('wordnet')
stopword = nltk.corpus.stopwords.words('english')
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


# emoticons funtion
def convert_emoticons(text):
    for emoti in emot.EMOTICONS:
        text = re.sub(u'(' + emoti + ')', "_".join(emot.EMOTICONS[emoti].replace(",", "").split()), text)
    return text

# remove handles function
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt

# remove special characters and punctuations
def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

# tokenisation function

def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

# remove stopwords
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text

# lemmatizer function
wn = nltk.WordNetLemmatizer()
def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

corpus = []
for i in range(0,len(train)):
    tweet_val = train["tweet"][i]
    # lowercase of tweet column
    tweet_val = tweet_val.lower()
    # remove emoticons
    tweet_val = convert_emoticons(str(tweet_val))
    tweet_val = str(tweet_val)
    tweet_val = np.vectorize(remove_pattern)(tweet_val, "@[\w]*")
    tweet_val = np.vectorize(remove_pattern)(tweet_val, "#[\w]*")
    tweet_val = str(tweet_val)
    tweet_val = re.sub(r'[^\w\s]', ' ', tweet_val)
    tweet_val = remove_punct(tweet_val)
    tweet_val = tweet_val.replace("[^a-zA-Z#]", " ")
    tweet_val = tweet_val.lower()
    tweet_val = tweet_val.replace("sxsw", " ")
    tweet_val = tweet_val.replace("link", " ")
    tweet_val = tweet_val.replace("quot", " ")
    tweet_val = tweet_val.replace("amp", " ")
    tweet_val = tweet_val.replace("google", " ")
    tweet_val = tweet_val.replace("ipad", " ")
    tweet_val = tweet_val.replace("iphone", " ")
    tweet_val = tweet_val.replace("apple", " ")
    tweet_val = tweet_val.replace("app", " ")
    tweet_val = tweet_val.replace("called", " ")
    tweet_val = tweet_val.replace("austin", " ")
    tweet_val = tweet_val.replace("store", " ")
    tweet_val = tweet_val.replace("android", " ")
    tweet_val = tweet_val.replace("pop", " ")
    tweet_val = tweet_val.replace("via", " ")
    tweet_val = tweet_val.replace("social", " ")
    tweet_val = tweet_val.replace("network", " ")
    tweet_val = tweet_val.replace("mention", " ")
    tweet_val = tweet_val.replace("new", " ")
    tweet_val = ' '.join([w for w in tweet_val.split() if len(w) > 2])
    tweet_val = tokenize(tweet_val.lower())
    tweet_val = remove_stopwords(tweet_val)
    tweet_val = lemmatizing(tweet_val)
    tweet_val = ' '.join(tweet_val)
    corpus.append(tweet_val)

print(len(corpus))
# tfidf = TfidfVectorizer()
# X = tfidf.fit_transform(corpus).toarray()
# pickle.dump(tfidf, open('tfidf.pkl','wb'))
cv = CountVectorizer(max_features=5250)
X = cv.fit_transform(corpus).toarray()
pickle.dump(cv, open('cv.pkl','wb'))
y =  train['sentiment'].values
print(X.shape)
print(y.shape)


print("*"*50)
print("Dividing the dataset into train and test")
print("*"*50)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

# print("*"*50)
# print("Apply Naiive Bayes classification")
# print("*"*50)
# nb = MultinomialNB(alpha=0.1)
# nb.fit(X_train,y_train)
# y_pred = nb.predict(X_test)
# print(nb.score(X,y))
# cm = confusion_matrix(y_test,y_pred)
# cr = classification_report(y_test,y_pred)
# print(cm)
# print(cr)

# print("*"*50)
# print("Apply Naiive Bayes classification")
# print("*"*50)
# gaussian_nb = GaussianNB()
# gaussian_nb.fit(X_train,y_train)
# y_pred = gaussian_nb.predict(X_test)
# print(gaussian_nb.score(X,y))
# cm = confusion_matrix(y_test,y_pred)
# cr = classification_report(y_test,y_pred)
# print(cm)
# print(cr)

print("*"*50)
print("Apply Logistic Regression")
print("*"*50)
# log_reg = LogisticRegression(random_state=42,class_weight='balanced',C=1000,max_iter=500)
log_reg = LogisticRegression(random_state=42,class_weight='balanced',C=500,max_iter=500)
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)
print(log_reg.score(X,y))
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
print(cm)
print(cr)

# print("*"*50)
# print("Executing Random Forest classifier")
# print("*"*50)
# rfc = RandomForestClassifier(random_state=42)
# rfc.fit(X_train,y_train)
# y_pred = rfc.predict(X_test)
# rfc.score(X_test,y_test)
# cm = confusion_matrix(y_test,y_pred)
# cr = classification_report(y_test,y_pred)
# print(rfc.score(X,y))
# print(cm)
# print(cr)

# Saving model to disk
log_reg_final = LogisticRegression(random_state=42)
log_reg_final.fit(X,y)

# rfc = RandomForestClassifier(random_state=42)
# rfc.fit(X,y)
pickle.dump(log_reg_final, open('TwitterSentimentAnalysisFinalModel.pkl','wb'))
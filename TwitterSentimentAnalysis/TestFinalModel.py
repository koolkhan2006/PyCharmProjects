
import numpy as np
import pandas as pd
import warnings
import emot
import string
import pickle
import re
import nltk

warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
cv = pickle.load(open('cv.pkl', 'rb'))
model = pickle.load(open('TwitterSentimentAnalysisFinalModel.pkl', 'rb'))
# cv = pickle.load(open('tfidf.pkl', 'rb'))
# model = pickle.load(open('TwitterSentimentAnalysisFinalModel.pkl', 'rb'))

test = pd.read_csv("test.csv", encoding = "Latin - 1")

# removing null or blank data row from test data set
print("*" * 50)
print("Check count for missing values in each column and drop that rows")
print("*" * 50)
print(test.isnull().sum())
test = test.dropna()
print(test.isnull().sum())
test = test.reset_index(drop=True)
print(test.shape)

print(test.shape)

# lowercase of tweet column
test["tweet"] = test["tweet"].str.lower()
# checking datatype of tweet column
print(test.dtypes)
test['tweet'] = test['tweet'].astype(str)
# changing column width to get full display
pd.set_option('display.max_colwidth', -1)


# emoticons funtion
def convert_emoticons(text):
    for emoti in emot.EMOTICONS:
        text = re.sub(u'(' + emoti + ')', "_".join(emot.EMOTICONS[emoti].replace(",", "").split()), text)
    return text

# remove emoticons
test['tweet'] = test['tweet'].apply(lambda x: convert_emoticons(str(x)))


# remove handles function
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)

    return input_txt


test['tweet'] = test['tweet'].astype(str)
test['tweet'] = np.vectorize(remove_pattern)(test['tweet'], "@[\w]*")
test['tweet'] = np.vectorize(remove_pattern)(test['tweet'], "#[\w]*")


# remove special characters and punctuations
def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct


test['tweet'] = test['tweet'].astype(str)
test['tweet'] = test['tweet'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
test['tweet'] = test['tweet'].apply(lambda x: remove_punct(x))
test['tweet'] = test['tweet'].str.replace("[^a-zA-Z#]", " ")

# removing most important hashtags
test['tweet'] = test['tweet'].str.lower()
test['tweet'] = test['tweet'].str.replace("sxsw", " ")
test['tweet'] = test['tweet'].str.replace("link", " ")
test['tweet'] = test['tweet'].str.replace("quot", " ")
test['tweet'] = test['tweet'].str.replace("amp", " ")
test['tweet'] = test['tweet'].str.replace("google", " ")
test['tweet'] = test['tweet'].str.replace("ipad", " ")
test['tweet'] = test['tweet'].str.replace("iphone", " ")
test['tweet'] = test['tweet'].str.replace("apple", " ")
test['tweet'] = test['tweet'].str.replace("app", " ")
test['tweet'] = test['tweet'].str.replace("called", " ")
test['tweet'] = test['tweet'].str.replace("austin", " ")
test['tweet'] = test['tweet'].str.replace("store", " ")
test['tweet'] = test['tweet'].str.replace("android", " ")
test['tweet'] = test['tweet'].str.replace("pop", " ")
test['tweet'] = test['tweet'].str.replace("via", " ")
test['tweet'] = test['tweet'].str.replace("social", " ")
test['tweet'] = test['tweet'].str.replace("network", " ")
test['tweet'] = test['tweet'].str.replace("mention", " ")
test['tweet'] = test['tweet'].str.replace("new", " ")

# remove words with less than 2 characters
test['tweet'] = test['tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w) > 2]))


# tokenisation function

def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens


test['tweet'] = test['tweet'].apply(lambda x: tokenize(x.lower()))

# In[87]:
stopword = nltk.corpus.stopwords.words('english')


# remove stopwords
def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stopword]
    return text


test['tweet'] = test['tweet'].apply(lambda x: remove_stopwords(x))

# lemmatizer function
wn = nltk.WordNetLemmatizer()


def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text


test['tweet'] = test['tweet'].apply(lambda x: lemmatizing(x))

# Convert list to str
test_string = pd.Series([' '.join(x) for x in test['tweet']])

test['tweet'] = test_string

# Word cloud for all words
test['tweet'] = test['tweet'].astype(str)

corpus = []
for i in range(0, len(test)):
    corpus.append(test['tweet'][i])

X_test = cv.transform(corpus).toarray()
y_pred = model.predict(X_test)

submission = pd.read_csv('sample_submission.csv')
print(submission['sentiment'].shape)
submission['sentiment'] = y_pred

output=pd.DataFrame(submission)
output.to_csv(r"results.csv",index=False)
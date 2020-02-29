import pandas as pd
import os
import warnings
import re
import pickle

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
warnings.filterwarnings("ignore")
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

print("*"*50)
print("Read the tsv file")
print("*"*50)
df = pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
print(len(df))

print("*"*50)
print("1. Remove all numbers and punctuations")
print("2. Convert to lower case")
print("3. Split into word list")
print("4. Remove stopwords")
print("5. Stemming. To remove Sparsity")
print("6. Join the words")
print("*"*50)
nltk.download('stopwords')
corpus = []

for i in range(0,len(df)):
    review = re.sub('[^a-zA-z]',' ',df['Review'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words("english"))]
    ps = PorterStemmer()
    review = [ps.stem(y) for y in review if not y in set(stopwords.words("english"))]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
pickle.dump(cv, open('cv.pkl','wb'))
y =  df.iloc[:,1].values
print(X.shape)
print(y.shape)

print("*"*50)
print("Dividing the dataset into train and test")
print("*"*50)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

print("*"*50)
print("Apply Naiive Bayes classification")
print("*"*50)
nb = GaussianNB()
nb.fit(X_train,y_train)
y_pred = nb.predict(X_test)
print(nb.score(X,y))
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)
print(cm)
print(cr)

# Saving model to disk
pickle.dump(nb, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
# intialise data of lists.
data = {'Review': ['Not tasty and the texture was just nasty']}
df_value = pd.DataFrame(data)
corpus_value = []
for i in range(0,len(df_value)):
    review = re.sub('[^a-zA-z]',' ',df_value['Review'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words("english"))]
    ps = PorterStemmer()
    review = [ps.stem(y) for y in review if not y in set(stopwords.words("english"))]
    review = ' '.join(review)
    corpus_value.append(review)

X_value = cv.transform(corpus_value).toarray()
print(model.predict(X_value))
import pandas as pd
import warnings
import re
import pickle

warnings.filterwarnings("ignore")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
# cv = pickle.load(open('cv.pkl', 'rb'))
# model = pickle.load(open('TwitterSentimentAnalysisModel.pkl', 'rb'))
cv = pickle.load(open('cv1.pkl', 'rb'))
model = pickle.load(open('TwitterSentimentAnalysisModel1.pkl', 'rb'))

print("*"*50)
print("Read the test csv file")
print("*"*50)
df_test = pd.read_csv("test.csv")
print(len(df_test))

print("*"*50)
print("Check count for missing values in test in each column and drop that rows")
print("*"*50)
print(df_test.isnull().sum())
df = df_test.dropna()
print(df_test.isnull().sum())
df_test = df_test.reset_index(drop=True)
print(df_test.shape)

corpus_value = []
for i in range(0,len(df_test)):
    review = re.sub('[^a-zA-z]',' ',df_test['tweet'][i])
    review = review.lower()
    review = review.split()
    review = [word for word in review if not word in set(stopwords.words("english"))]
    ps = PorterStemmer()
    review = [ps.stem(y) for y in review if not y in set(stopwords.words("english"))]
    review = ' '.join(review)
    corpus_value.append(review)

X_test = cv.transform(corpus_value).toarray()
y_pred = model.predict(X_test)

submission = pd.read_csv('sample_submission.csv')
print(submission['sentiment'].shape)
submission['sentiment'] = y_pred

output=pd.DataFrame(submission)
output.to_csv(r"results.csv",index=False)
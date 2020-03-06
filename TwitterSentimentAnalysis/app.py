import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_bootstrap import Bootstrap
import pickle
import pandas as pd
import re
import nltk
import string
import emot

app = Flask(__name__)
model = pickle.load(open('TwitterSentimentAnalysisFinalModel.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))
nltk.download('stopwords')
stopword = nltk.corpus.stopwords.words('english')
Bootstrap(app)
@app.route('/')
def home():
    return render_template('indexbootstrap.html')

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

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    tweet = [x for x in request.form.values()]
    data = {'tweet': tweet}
    df_value = pd.DataFrame(data)
    corpus_value = []
    for i in range(0, len(df_value)):
        tweet_val = df_value["tweet"][i]
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
        corpus_value.append(tweet_val)

    X_value = cv.transform(corpus_value).toarray()
    prediction = model.predict(X_value)

    if(prediction[0] == 0):
        output = 'Negative'
    elif(prediction[0] == 1):
        output = 'Neutral'
    elif (prediction[0] == 2):
        output = 'Positive'
    else:
        output = 'Cant say'

    return render_template('indexbootstrap.html', prediction_text='Predicted sentiment for the review is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

# if __name__ == "__main__":
#     app.run(host='0.0.0.0',port=8080)
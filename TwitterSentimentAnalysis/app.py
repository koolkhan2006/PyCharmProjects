import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_bootstrap import Bootstrap
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))
nltk.download('stopwords')
Bootstrap(app)
@app.route('/')
def home():
    return render_template('indexbootstrap.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    review = [x for x in request.form.values()]
    data = {'Review': review}
    df_value = pd.DataFrame(data)
    corpus_value = []
    for i in range(0, len(df_value)):
        review = re.sub('[^a-zA-z]', ' ', df_value['Review'][i])
        review = review.lower()
        review = review.split()
        review = [word for word in review if not word in set(stopwords.words("english"))]
        ps = PorterStemmer()
        review = [ps.stem(y) for y in review if not y in set(stopwords.words("english"))]
        review = ' '.join(review)
        corpus_value.append(review)

    X_value = cv.transform(corpus_value).toarray()
    prediction = model.predict(X_value)

    if(prediction[0] == 0):
        output = 'Negative'
    else:
        output = 'Positive'

    return render_template('indexbootstrap.html', prediction_text='Predicted sentiment for the review is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)

# if __name__ == "__main__":
#     app.run(host='0.0.0.0',port=8080)
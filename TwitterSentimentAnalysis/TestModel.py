import pickle
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
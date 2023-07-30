
### this file generates the model.pkl(svm + knn) files that is latter loaded in app.py 

#load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df= pd.read_csv("dataset.csv") # already preprocessed dataset using kaggle notebook
df.drop(["index"],axis=1,inplace=True)

#split
x_train=df["feature"]
y_train=df["target"]

#text to number
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(x_train)

#model train
from sklearn.svm import SVC
model1 = SVC()
model1.fit(X_train, y_train)

# Create a KNN classifier
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=3)
model2.fit(X_train, y_train)

#saving the model in .pkl file
import joblib
model_filename = 'svm_model.pkl'
joblib.dump(model1, model_filename)
model_filename = 'knn_model.pkl'
joblib.dump(model2, model_filename)

#saving the vectorizer in .pkl file
vectorizer_filename = 'tfidf_vectorizer.pkl'
joblib.dump(vectorizer, vectorizer_filename)

import streamlit as st
from sentence_transformers import SentenceTransformer
from util import * # contains helper functions
#loading the models 
import joblib

st.header(" Med_API By PB")
query = False
options = st.radio(
    'Choose model',
    ('LR + transformer','SVM + tfidf','KNN + tfidf'))

if 'SVM + tfidf' in options:
    model1 = joblib.load('svm_model.pkl')
if 'KNN + tfidf' in options:
    model2 = joblib.load('knn_model.pkl')   
if 'LR + transformer' in options:
    model3 = joblib.load('LR_model.pkl')


query = st.text_input("Enter your question")#user input
button = st.button("Submit")


#tfidf vectorizer and transformer
vectorizer=joblib.load('tfidf_vectorizer.pkl')
model = SentenceTransformer('all-MiniLM-L6-v2')



# Transform the user input into its embedding for lR
text= spacy_tokenizer(query)
text= model.encode(text).reshape(1,-1)



if button and query:
    if 'SVM + tfidf' in options:
        with st.spinner("Predicting..."):# if select this option
            #pre-process text
            text= spacy_tokenizer(query)
            # Transform the user input into its TF-IDF representation for svm knn
            input_tfidf = vectorizer.transform([text])
            #predicting
            answer = model1.predict(input_tfidf)[0]
            #display
            st.success("Predicted Disease: "+answer)
    if 'KNN + tfidf' in options:
        with st.spinner("Predicting..."):# if select this option
            #pre-process text
            text= spacy_tokenizer(query)
            # Transform the user input into its TF-IDF representation for svm knn
            input_tfidf = vectorizer.transform([text])
            #predicting
            answer = model2.predict(input_tfidf)[0]
            #display
            st.success("Predicted Disease: "+answer)

    if 'LR + transformer' in options:# if select ask a question option
        with st.spinner("Searching for the answer..."):
            #pre-process text
            text= spacy_tokenizer(query)
            # Transform the user input into its embedding for lR
            text= model.encode(text).reshape(1,-1)
            #predicting
            dic= get_dict()
            i=model3.predict(text)[0]#(op is from 0-40 unique labels)
            answer = dic[i]#decode and get the disease
            #display
            st.success("Predicted Disease: "+answer)





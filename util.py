
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy# for text processing
import string
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
punctuations = string.punctuation

#Read
data= pd.read_csv("dataset.csv") # already preprocessed dataset using kaggle notebook
data.drop(["index"],axis=1,inplace=True)

# text pre-processing
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    doc = nlp(sentence)



    # print(doc)
    # print(type(doc))

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() for word in doc ]

    # print(mytokens)

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]
    
    sentence = " ".join(mytokens)
    # return preprocessed list of tokens
    return sentence

def get_dict(): # used by LR to decode the disease
 dic=data["target"].to_dict()#{0:"vertigo",...31:"malaria"...}
 return dic
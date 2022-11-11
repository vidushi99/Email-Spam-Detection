import streamlit as st
import pickle
from textblob import TextBlob
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

punct = string.punctuation
ps = PorterStemmer()
tfi = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # puctuation
    l = []
    for word in text:
        if word.isalnum():
            l.append(word)

    # stopword
    text = l[:]
    l.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in punct:
            l.append(i)

    # stemming
    text = l[:]
    l.clear()
    for i in text:
        l.append(ps.stem(i))
    text = " ".join(l)

    text = TextBlob(text)
    text = text.correct()
    return text.raw


st.title('Email/SMS Spam Detection')
input_msg = st.text_area('Enter The Message')
if st.button('Predict'):
    #preprocess
    transformed_text = transform_text(input_msg)
    #vectorize
    vector_input = vectorizer.transform([transformed_text])
    #prediction
    output = tfi.predict(vector_input)[0]
    if output==1:
        st.header('Spam')
    else:
        st.header("Not Spam")
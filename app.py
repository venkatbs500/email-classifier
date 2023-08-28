import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer  # Add this import
import string

# Create a PorterStemmer instance
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()  # converts into lower case
    text = nltk.word_tokenize(text)  # tokenizes the sentence

    # Remove special characters and keep only alphanumeric words
    y = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuation
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("enter a message")

if st.button("Predict"):

    #1.preprocess
    transformed_sms = transform_text(input_sms)
    #2.vectorize
    vector_input = tfidf.transform([transformed_sms])
    #3.predict
    result = model.predict(vector_input)[0]
    #4.display
    if result == 1:
        st.header("This message is A SPAM message")
    else:
        st.header("This message is NOT A SPAM message" )


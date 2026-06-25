import streamlit as st
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Load model

vectors = joblib.load(open("../reports/models/vectors.pkl", "rb"))
model = joblib.load(open("../reports/models/model.pkl", "rb"))

# Preprocess

def transform(text):
    
    # Lowercase
    text = text.lower()
    
    # Tokenization
    tokens = nltk.word_tokenize(text)
    
    # Remove non-alphanumeric tokens
    tokens = [word for word in tokens if word.isalnum()]
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Stemming
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    
    return " ".join(tokens)

st.title("Spam Email/Message Detector")

input_sms = st.text_area(label='Enter your message to detect ham/spam.')

if st.button('Detect'):
    
    transformed_sms = transform(input_sms)

    #  Vectorize
    vector_input = vectors.transform([transformed_sms])

    # Predict

    prediction = model.predict(vector_input)[0]

    # Display

    if prediction == 1:
        st.header("Spam Detected")
        
    else:
        st.header("No Spam Detected")

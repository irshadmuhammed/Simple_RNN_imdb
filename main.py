import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Embedding,SimpleRNN
from tensorflow.keras.datasets import imdb

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key,value in word_index.items()}

model = load_model('rnn_model_imdb.h5')


def decod_reviw(encoded_reviw):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_reviw])

def preprocess_text(text):
    words = text.lower().split()
    ecoded_reviw = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([ecoded_reviw],maxlen=500)
    return padded_review

## stream lit
import streamlit as st

st.title('Sentiment Analysis from IMDB')
st.write("Enter the reviw in the box below")

user_input = st.text_area("Review")

if st.button("classify"):
    processed_input = preprocess_text(user_input)
    prediction = model.predict(processed_input)
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'

    st.write(f'The sentiment of the review is {sentiment} with a confidence of {prediction[0][0]}')
else:
    st.write("Please enter a review to classify")   
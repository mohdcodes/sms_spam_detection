# app/main.py
import streamlit as st
import pandas as pd
import joblib

# Load model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('../models/spam_detector.pkl')

model = load_model()

# Streamlit app
st.title('SMS Spam Detection')
sms_text = st.text_area('Enter SMS text')

if st.button('Predict'):
    if sms_text:
        prediction = model.predict([sms_text])[0]
        label = 'Spam' if prediction == 1 else 'Not Spam'
        st.write(f'The message is: {label}')
    else:
        st.write('Please enter an SMS text.')

import streamlit as st
import sys
import os

# Ensure backend.py is found
sys.path.append(os.path.dirname(__file__))

from backend import predict_word

st.title("Next Word Prediction (Deep Learning)")
st.caption("LSTM-based NLP Model | By Nadish")

text = st.text_input("Enter at least two words")

if st.button("Predict"):
    if len(text.split()) < 2:
        st.warning("Please enter at least two words")
    else:
        result = predict_word(text)
        st.success(f"Next Word: {result}")

import streamlit as st
from backend import predict_word

st.title("Next Word Prediction (Deep Learning)")
st.caption("LSTM-based NLP Model | By Nadish")

text = st.text_input("Enter text")

if st.button("Predict"):
    if len(text.split()) < 2:
        st.warning("Enter at least two words")
    else:
        result = predict_word(text)
        st.success(f"Next Word: {result}")

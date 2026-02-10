
import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.title("Next Word Prediction System")

user_input = st.text_input("Enter some text:", "")

if st.button("Predict Next Word"):
    if user_input.strip():
        response = requests.get(API_URL, params={"text": user_input, "top_k": 5})
        if response.status_code == 200:
            result = response.json()
            words = result.get("next_words", [])
            st.write("Top predictions:")
            for w in words:
                st.write(f"- {w}")
        else:
            st.error("Error calling API")

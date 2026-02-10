
import streamlit as st
import requests

# Replace with your backend URL if deployed remotely
API_URL = "http://localhost:8000/predict"

st.title("📝 Next Word Prediction System")

# Input text field
user_input = st.text_input("Enter some text:", "")

# Number of predictions to show
top_k = st.slider("Number of predictions", min_value=1, max_value=10, value=5)

# Predict button
if st.button("Predict Next Word"):
    if user_input.strip():
        try:
            response = requests.get(API_URL, params={"text": user_input, "top_k": top_k})
            if response.status_code == 200:
                result = response.json()
                words = result.get("next_words", [])
                if words:
                    st.subheader("Top predictions:")
                    for i, w in enumerate(words, start=1):
                        st.write(f"{i}. {w}")
                else:
                    st.warning("No predictions available.")
            else:
                st.error(f"Backend

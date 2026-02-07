
import streamlit as st
import requests
import os

st.set_page_config(page_title="PKR Word Predictor", page_icon="🇵🇰")

st.title("🇵🇰 PKR Next Word Predictor")
st.info("Assignment: FastAPI + Streamlit Client-Server Architecture")

# Input section
text_input = st.text_input("Enter text to predict the next word:", placeholder="e.g. The State Bank of")

if st.button("Predict"):
    if text_input:
        with st.spinner("Requesting prediction from FastAPI backend..."):
            try:
                # In production, set BACKEND_URL to your deployed FastAPI address
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8000/predict")
                response = requests.post(backend_url, json={"text": text_input})
                
                if response.status_code == 200:
                    data = response.json()
                    preds = data.get("predictions", [])
                    
                    if not preds:
                        st.warning("No predictions returned.")
                    else:
                        st.subheader("Results")
                        for p in preds:
                            with st.expander(f"Word: **{p['word']}** ({int(p['probability']*100)}%)"):
                                st.write(f"**Reasoning:** {p['reasoning']}")
                else:
                    st.error(f"Error from backend: {response.text}")
            except Exception as e:
                st.error(f"Could not connect to backend at {backend_url}. Make sure main.py is running.")
    else:
        st.warning("Please enter some text.")

st.sidebar.markdown("""
### How to Run:
1. **Start Backend:** `python main.py`
2. **Start Frontend:** `streamlit run app.py`
""")

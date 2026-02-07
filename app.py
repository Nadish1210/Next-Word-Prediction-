import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import re
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="Next Word Prediction System",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .prediction-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .word-badge {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.2rem;
        display: inline-block;
    }
    </style>
""", unsafe_allow_html=True)

# Global variables
seq_length = 5

@st.cache_resource
def load_model_and_tokenizer():
    """Load the trained model and tokenizer"""
    try:
        model = tf.keras.models.load_model('next_word_model.h5')
        with open('tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def preprocess_text(text):
    """Clean and preprocess the text"""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text

def predict_next_words(text: str, model, tokenizer, top_k: int = 5):
    """Get next word predictions"""
    try:
        # Preprocess input text
        cleaned_text = preprocess_text(text)
        
        if not cleaned_text:
            return None
        
        # Tokenize the input text
        tokens = tokenizer.texts_to_sequences([cleaned_text])[0]
        
        if len(tokens) == 0:
            return None
        
        # Take the last seq_length tokens
        if len(tokens) < seq_length:
            tokens = [0] * (seq_length - len(tokens)) + tokens
        else:
            tokens = tokens[-seq_length:]
        
        # Prepare input for prediction
        input_seq = np.array([tokens])
        
        # Get predictions
        predictions = model.predict(input_seq, verbose=0)[0]
        
        # Get top k predictions
        top_k = min(top_k, len(predictions))
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        # Create reverse word index
        reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
        
        # Prepare results
        results = []
        for idx in top_indices:
            if idx > 0 and idx in reverse_word_index:
                word = reverse_word_index[idx]
                probability = float(predictions[idx])
                results.append({
                    "word": word,
                    "probability": probability,
                    "confidence": f"{probability * 100:.2f}%"
                })
        
        return results
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def generate_sentence(text: str, model, tokenizer, num_words: int = 10):
    """Generate a complete sentence"""
    try:
        cleaned_text = preprocess_text(text)
        generated_text = cleaned_text
        
        for _ in range(num_words):
            tokens = tokenizer.texts_to_sequences([generated_text])[0]
            
            if len(tokens) < seq_length:
                tokens = [0] * (seq_length - len(tokens)) + tokens
            else:
                tokens = tokens[-seq_length:]
            
            input_seq = np.array([tokens])
            predictions = model.predict(input_seq, verbose=0)[0]
            
            predicted_index = np.argmax(predictions)
            
            if predicted_index == 0:
                break
            
            reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
            if predicted_index in reverse_word_index:
                predicted_word = reverse_word_index[predicted_index]
                generated_text += " " + predicted_word
        
        return generated_text
    
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        return None

def main():
    # Load model
    model, tokenizer = load_model_and_tokenizer()
    
    # Header
    st.title("🔮 Next Word Prediction System")
    st.markdown("### Powered by NLP & Deep Learning")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model Status
        if model is not None and tokenizer is not None:
            st.success("✅ Model Loaded")
            st.info(f"**Vocabulary Size:** {len(tokenizer.word_index) + 1}")
            st.info(f"**Sequence Length:** {seq_length}")
            st.info(f"**Model Type:** LSTM")
        else:
            st.error("❌ Model Not Loaded")
            st.warning("Please upload model files:\n- next_word_model.h5\n- tokenizer.pkl")
            return
        
        st.markdown("---")
        
        # Mode selection
        mode = st.radio(
            "Select Mode:",
            ["Next Word Prediction", "Sentence Generation"]
        )
        
        st.markdown("---")
        
        if mode == "Next Word Prediction":
            top_k = st.slider("Number of predictions:", 1, 10, 5)
        else:
            num_words = st.slider("Words to generate:", 5, 20, 10)
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This system uses LSTM neural networks to predict the next word 
        based on the input context. It's trained on Pakistani Rupee history data.
        """)
    
    # Mode: Next Word Prediction
    if mode == "Next Word Prediction":
        st.header("🎯 Next Word Prediction")
        st.markdown("Enter some text and get predictions for the next word:")
        
        # Input
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.text_input(
                "Enter text:",
                placeholder="e.g., The Pakistani rupee is",
                key="input_text"
            )
        with col2:
            predict_btn = st.button("🔍 Predict", type="primary")
        
        # Example texts
        st.markdown("**Quick Examples:**")
        examples = [
            "The Pakistani rupee is",
            "coins were introduced in",
            "the currency was",
            "State Bank of",
            "legal tender in"
        ]
        
        example_cols = st.columns(len(examples))
        for idx, (col, example) in enumerate(zip(example_cols, examples)):
            with col:
                if st.button(f"📝 {example[:15]}...", key=f"example_{idx}"):
                    st.session_state.input_text = example
                    st.rerun()
        
        # Prediction
        if predict_btn and user_input:
            with st.spinner("Predicting..."):
                result = predict_next_words(user_input, model, tokenizer, top_k)
                
                if result:
                    st.success("✅ Predictions Generated!")
                    
                    st.markdown("### 📊 Predictions:")
                    
                    # Display predictions
                    for idx, pred in enumerate(result, 1):
                        col1, col2, col3 = st.columns([1, 3, 2])
                        
                        with col1:
                            st.markdown(f"**#{idx}**")
                        with col2:
                            st.markdown(f"<span class='word-badge'>{pred['word']}</span>", 
                                      unsafe_allow_html=True)
                        with col3:
                            confidence = pred['confidence']
                            st.progress(float(pred['probability']))
                            st.caption(f"Confidence: {confidence}")
                        
                        st.markdown("---")
                    
                    # Show complete sentence preview
                    st.markdown("### 📝 Preview:")
                    for pred in result[:3]:
                        preview_text = f"{user_input} **{pred['word']}**"
                        st.markdown(f"➤ {preview_text}")
        
        elif predict_btn:
            st.warning("⚠️ Please enter some text first!")
    
    # Mode: Sentence Generation
    else:
        st.header("📝 Sentence Generation")
        st.markdown("Start with some text and let the AI complete it:")
        
        # Input
        col1, col2 = st.columns([3, 1])
        with col1:
            user_input = st.text_area(
                "Enter starting text:",
                placeholder="e.g., The Pakistani rupee",
                height=100,
                key="gen_input"
            )
        with col2:
            generate_btn = st.button("🚀 Generate", type="primary")
        
        # Generation
        if generate_btn and user_input:
            with st.spinner("Generating sentence..."):
                result = generate_sentence(user_input, model, tokenizer, num_words)
                
                if result:
                    st.success("✅ Sentence Generated!")
                    
                    st.markdown("### 📄 Results:")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Original Text:**")
                        st.info(user_input)
                    
                    with col2:
                        st.markdown("**Words Added:**")
                        st.success(num_words)
                    
                    st.markdown("### ✨ Generated Text:")
                    st.markdown(f"<div class='prediction-card'>{result}</div>", 
                              unsafe_allow_html=True)
                    
                    # Download option
                    st.download_button(
                        label="📥 Download Generated Text",
                        data=result,
                        file_name="generated_text.txt",
                        mime="text/plain"
                    )
        
        elif generate_btn:
            st.warning("⚠️ Please enter some text first!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>Next Word Prediction System | Built with Streamlit & TensorFlow</p>
            <p>NLP Assignment - 2024</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

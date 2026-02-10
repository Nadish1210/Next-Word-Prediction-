
import os
import pickle
import numpy as np
import re
from fastapi import FastAPI
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_FILE = "lstm_model.h5"
TOKENIZER_FILE = "tokenizer.pkl"
DATA_FILE = "data.txt"

api = FastAPI()

def train_model():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read().lower()

    text = re.sub(r'[^a-z\s]', '', text)

    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts([text])

    sequences = []
    for line in text.splitlines():
        tokens = tokenizer.texts_to_sequences([line])[0]
        for i in range(2, len(tokens)):
            sequences.append(tokens[:i+1])

    max_len = max(len(seq) for seq in sequences)
    sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')

    X = sequences[:, :-1]
    y = sequences[:, -1]

    vocab_size = len(tokenizer.word_index) + 1

    model = Sequential([
        Embedding(vocab_size, 64, input_length=max_len-1),
        LSTM(100),
        Dense(vocab_size, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=15, verbose=0)

    model.save(MODEL_FILE)
    with open(TOKENIZER_FILE, "wb") as f:
        pickle.dump((tokenizer, max_len), f)

    return model, tokenizer, max_len

if os.path.exists(MODEL_FILE):
    model = load_model(MODEL_FILE)
    tokenizer, max_len = pickle.load(open(TOKENIZER_FILE, "rb"))
else:
    model, tokenizer, max_len = train_model()

def predict_word(text, top_k=5):
    seq = tokenizer.texts_to_sequences([text.lower()])[0]
    seq = pad_sequences([seq], maxlen=max_len-1, padding='pre')
    pred = model.predict(seq, verbose=0)[0]

    top_indices = np.argsort(pred)[-top_k:][::-1]
    words = [tokenizer.index_word.get(i, "") for i in top_indices]
    return words

@api.get("/predict")
def predict_api(text: str, top_k: int = 5):
    return {"next_words": predict_word(text, top_k)}

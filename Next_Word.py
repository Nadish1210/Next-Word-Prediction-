
import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from typing import List

app = FastAPI(title="NLP Next-Word Prediction API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use gemini-3-flash-preview for NLP tasks
MODEL_NAME = 'gemini-3-flash-preview'

# The full corpus for the model to use as context
CORPUS = """
The Pakistani rupee (Urdu: روپیہ; ISO code: PKR; symbol: 𞱱; abbreviation: Re (singular) and Rs (plural)) is the official currency of the Islamic Republic of Pakistan. It is divided into one hundred paise (Urdu: پیسہ); however, paisa-denominated coins have not been legal tender since 2013. The issuance of the currency is controlled by the State Bank of Pakistan. It was officially adopted by the Government of Pakistan in 1949. Earlier the coins and notes were issued and controlled by the Reserve Bank of India until 1949, when it was handed over to the Government and State Bank of Pakistan, by the Government and Reserve Bank of India.

In Pakistani English, large values of rupees are counted in thousands; lac (hundred thousands); crore (ten-millions); arab (billion); kharab (hundred billion). Numbers are still grouped in thousands.

History
The word rūpiya is derived from the Sanskrit word rūpya, which means "wrought silver, a coin of silver". Rūpaya was used to denote the coin introduced by Sher Shah Suri during his reign from 1540 to 1545 CE. The Pakistan (Monetary System and Reserve Bank) Order, 1947 was issued on 14 August 1947. It designated the Reserve Bank of India (RBI) as the temporary monetary authority for both India and Pakistan until 30 September 1948.

In January 1961, the currency was decimalised, with the rupee subdivided into 100 pice, renamed paise later the same year. Coins denominated in paise have not been issued since 1996. In 1972, Bangladesh introduced the taka at parity with the Pakistani rupee. 

Coins for 1, 5 and 10 pice were issued in 1961. Re. 1/- coins were reintroduced in 1979, followed by Rs. 2/- in 1998 and Rs. 5/- in 2002. In 2016 a Rs. 10/- coin was introduced. In 2019 a commemorative Rs. 50/- coin was introduced to celebrate Guru Nanak.
"""

class PredictionRequest(BaseModel):
    text: str

class Prediction(BaseModel):
    word: str
    probability: float
    reasoning: str

class PredictionResponse(BaseModel):
    predictions: List[Prediction]

@app.post("/predict", response_model=PredictionResponse)
async def predict_next_word(request: PredictionRequest):
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="API_KEY not configured on server")

    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        
        prompt = f"""
        System: You are an NLP Next Word Predictor. Use the corpus below for context.
        Corpus: {CORPUS}
        
        Task: Predict the most likely next word for the input string.
        Return ONLY valid JSON.
        
        Input: "{request.text}"
        """
        
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "predictions": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "word": {"type": "string"},
                                    "probability": {"type": "number"},
                                    "reasoning": {"type": "string"}
                                },
                                "required": ["word", "probability", "reasoning"]
                            }
                        }
                    },
                    "required": ["predictions"]
                }
            }
        )
        
        return json.loads(response.text)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

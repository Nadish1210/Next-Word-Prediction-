
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

# Configuration
MODEL_NAME = 'gemini-3-flash-preview'

# The Pakistani Rupee Corpus for context
CORPUS = """
The Pakistani rupee (Urdu: روپیہ; ISO code: PKR; symbol: 𞱱; abbreviation: Re (singular) and Rs (plural)) is the official currency of the Islamic Republic of Pakistan. It is divided into one hundred paise (Urdu: پیسہ); however, paisa-denominated coins have not been legal tender since 2013. The issuance of the currency is controlled by the State Bank of Pakistan. It was officially adopted by the Government of Pakistan in 1949. Earlier the coins and notes were issued and controlled by the Reserve Bank of India until 1949, when it was handed over to the Government and State Bank of Pakistan, by the Government and Reserve Bank of India.

In Pakistani English, large values of rupees are counted in thousands; lac (hundred thousands); crore (ten-millions); arab (billion); kharab (hundred billion). Numbers are still grouped in thousands.

The word rūpiya is derived from the Sanskrit word rūpya, which means "wrought silver, a coin of silver". It is derived from the noun rūpa "shape, likeness, image". Rūpaya was used to denote the coin introduced by Sher Shah Suri during his reign from 1540 to 1545 CE.

The Pakistan (Monetary System and Reserve Bank) Order, 1947 was issued on 14 August 1947. It designated the Reserve Bank of India (RBI) as the temporary monetary authority for both India and Pakistan until 30 September 1948. In January 1961, the currency was decimalised, with the rupee subdivided into 100 pice, renamed paise later the same year. 

In 1948, coins were introduced in denominations of 1 pice, 1/2, 1 and 2 annas, 1/4, 1/2 and 1 rupee. Re. 1/- coins were reintroduced in 1979, followed by Rs. 2/- in 1998 and Rs. 5/- in 2002. In 2019 the Pakistan government introduced a commemorative Rs. 50/- coin to celebrate the 550th birthday of Guru Nanak.
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
        raise HTTPException(status_code=500, detail="API_KEY not found. Please set it in environment variables or Streamlit Secrets.")

    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        
        system_instruction = f"""
        You are an NLP model for Next Word Prediction.
        Context: {CORPUS}
        
        Instructions:
        Predict the top 3-5 most likely next words for the user's input.
        Return strictly valid JSON format.
        """
        
        response = model.generate_content(
            f"{system_instruction}\n\nInput text: '{request.text}'",
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
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Use 8000 for local dev, or the PORT env var for cloud deployment
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

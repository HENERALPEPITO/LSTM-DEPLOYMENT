import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import numpy as np
import asyncio
from hypercorn.asyncio import serve
from hypercorn.config import Config

# Import the LSTMPredictor class from lstm.py
from lstm import LSTMPredictor

app = FastAPI()

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict")
async def predict():
    try:
        # Load pre-trained model and scalers
        model, scaler_features, scaler_target = LSTMPredictor.load_model_and_scalers()
        
        # Create a predictor instance (without training)
        predictor = LSTMPredictor()
        predictor.model = model
        predictor.scaler_features = scaler_features
        predictor.scaler_target = scaler_target
        
        # Predict next 5 cases
        predictions = predictor.predict_next_cases()
        
        return {"predictions": predictions.tolist()}
    
    except Exception as e:
        return {"error": str(e)}
        
if __name__ == "__main__":
    config = Config()
    config.bind = ["127.0.0.1:8000"]
    asyncio.run(serve(app, config))
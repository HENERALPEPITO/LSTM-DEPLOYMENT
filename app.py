import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

import tensorflow as tf
import joblib
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Wrap model and scaler loading in try-except to catch initialization errors
try:
    # Load the pre-trained model and scalers
    model = tf.keras.models.load_model('lstm_model.h5')
    scaler_target = joblib.load('scaler_target.pkl')
    scaler_features = joblib.load('scaler_features.pkl')
except Exception as e:
    print(f"Error loading model or scalers: {e}")
    print(traceback.format_exc())
    model = None
    scaler_target = None
    scaler_features = None

# Define the window size used during training
WINDOW_SIZE = 10

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Verify model and scalers are loaded
        if model is None or scaler_target is None or scaler_features is None:
            raise ValueError("Model or scalers not properly initialized")

        # Read the uploaded CSV file
        contents = await file.read()
        
        # Print the first few bytes to debug file reading
        print(f"File contents (first 100 bytes): {contents[:100]}")
        
        # Try reading with different methods
        try:
            df = pd.read_csv(pd.iotools.common.StringIO(contents.decode('utf-8')))
        except Exception as utf8_error:
            print(f"UTF-8 decoding error: {utf8_error}")
            try:
                df = pd.read_csv(pd.iotools.common.StringIO(contents.decode('latin-1')))
            except Exception as latin1_error:
                print(f"Latin-1 decoding error: {latin1_error}")
                raise
        
        # Print DataFrame head for debugging
        print("DataFrame head:")
        print(df.head())
        print("\nDataFrame columns:", df.columns)
        
        # Preprocess the data (similar to training preprocessing)
        df['Time'] = pd.to_datetime(df['Time'] + '-1', format='%Y-w%W-%w')
        
        # Select features
        features = df[['Rainfall', 'Temperature', 'Humidity']]
        
        # Normalize features
        normalized_features = scaler_features.transform(features)
        
        # Create sequences
        X = []
        for i in range(WINDOW_SIZE, len(normalized_features)):
            X.append(normalized_features[i-WINDOW_SIZE:i])
        X = np.array(X)
        
        # Predict
        predictions_normalized = model.predict(X)
        
        # Inverse transform predictions
        predictions = scaler_target.inverse_transform(predictions_normalized)
        
        # Prepare response
        result = {
            "predictions": predictions.flatten().tolist(),
            "actual_dates": df['Time'][WINDOW_SIZE:].dt.strftime('%Y-%m-%d').tolist()
        }
        
        return JSONResponse(content=result)
    
    except Exception as e:
        print("Full error traceback:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
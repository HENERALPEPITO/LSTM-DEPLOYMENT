import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
import joblib

class LSTMPredictor:
    def __init__(self, file_path='completefinaldatasets.csv', window_size=3):
        self.file_path = file_path
        self.window_size = window_size
        self.model = None
        self.scaler_features = None
        self.scaler_target = None

    def load_and_preprocess_data(self):
        data = pd.read_csv(self.file_path)

        # Select features and target
        features = data[['Rainfall', 'Temperature', 'Humidity']]
        target = data['Cases']

        # Initialize scalers
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()

        # Normalize features and target
        normalized_features = self.scaler_features.fit_transform(features)
        normalized_target = self.scaler_target.fit_transform(target.values.reshape(-1, 1))

        # Combine normalized features and target
        normalized_data = np.hstack((normalized_features, normalized_target))

        return normalized_data

    def create_sequences(self, data):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i-self.window_size:i, :-1])  # All features except last column
            y.append(data[i, -1])  # Target (last column)
        return np.array(X), np.array(y)

    def build_lstm_model(self, input_shape):
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(64, activation='relu', return_sequences=False))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
        return model

    def train_model(self, epochs=50, batch_size=1):
        # Preprocess data
        normalized_data = self.load_and_preprocess_data()
        
        # Create sequences
        X, y = self.create_sequences(normalized_data)

        # Build and train model
        self.model = self.build_lstm_model(X.shape[1:])
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict_next_cases(self, steps=5):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")

        # Get the last sequence from the original data
        data = pd.read_csv(self.file_path)
        features = data[['Rainfall', 'Temperature', 'Humidity']]
        normalized_features = self.scaler_features.transform(features)

        # Take the last window_size rows
        last_sequence = normalized_features[-self.window_size:]
        
        # Predict next cases
        predictions = []
        current_sequence = last_sequence

        for _ in range(steps):
            # Reshape the sequence correctly
            input_sequence = current_sequence.reshape(1, self.window_size, -1)
            
            # Predict next value
            pred = self.model.predict(input_sequence)
            predictions.append(pred[0, 0])

            # Update current sequence by sliding window and appending prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, -1] = pred[0, 0]

        # Scale predictions back to original values
        predictions_scaled = self.scaler_target.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions_scaled.flatten()

    def save_model_and_scalers(self):
        # Save the trained model
        self.model.save('lstm_model.h5')
        
        # Save scalers
        joblib.dump(self.scaler_features, 'scaler_features.pkl')
        joblib.dump(self.scaler_target, 'scaler_target.pkl')
        print("Model and scalers saved successfully!")

    @classmethod
    def load_model_and_scalers(cls):
        # Load the model
        model = load_model('lstm_model.h5')
        
        # Load scalers
        scaler_features = joblib.load('scaler_features.pkl')
        scaler_target = joblib.load('scaler_target.pkl')
        
        return model, scaler_features, scaler_target

if __name__ == "__main__":
    # Create predictor instance
    predictor = LSTMPredictor()
    
    # Train the model
    predictor.train_model()
    
    # Save model and scalers
    predictor.save_model_and_scalers()
    
    # Predict next 5 cases
    predictions = predictor.predict_next_cases()
    print("Predicted next 5 cases:", predictions)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import joblib  # To save scaler

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data['Time'] = pd.to_datetime(data['Time'] + '-1', format='%Y-w%W-%w')

    features = data[['Rainfall', 'Temperature', 'Humidity']]
    target = data['Cases']

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    normalized_features = scaler_features.fit_transform(features)
    normalized_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

    normalized_data = pd.DataFrame(normalized_features, columns=features.columns, index=data.index)
    normalized_data['Cases'] = normalized_target

    return normalized_data, scaler_target, scaler_features

def create_sequences(data, target, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i])  # Collect 'window_size' rows
        y.append(target[i])  # Target at the end of the sequence
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # input_shape = (time_steps, features)
    model.add(LSTM(64, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=1):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    return history

def evaluate_model(model, X, y, scaler_target):
    y_pred = model.predict(X)
    predicted_actual_scale = scaler_target.inverse_transform(y_pred)
    y_actual_scale = scaler_target.inverse_transform(y)

    mse = mean_squared_error(y_actual_scale, predicted_actual_scale)
    rmse = np.sqrt(mse)

    return predicted_actual_scale.flatten(), y_actual_scale.flatten(), mse, rmse

def classify_predictions(predictions, actual_values, threshold=0.5):
    pred_class = (predictions >= threshold).astype(int)
    actual_class = (actual_values >= threshold).astype(int)

    accuracy = accuracy_score(actual_class, pred_class)
    precision = precision_score(actual_class, pred_class)
    recall = recall_score(actual_class, pred_class)

    return accuracy, precision, recall

if __name__ == "__main__":
    # Load and preprocess data
    data, scaler_target, scaler_features = load_and_preprocess_data('completefinaldatasets.csv')

    # Prepare sequences
    window_size = 10
    X, y = create_sequences(data.values, data['Cases'].values, window_size)

    # Split data
    split_index = int(len(X) * 0.85)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # Build and train model
    model = build_lstm_model(X_train.shape[1:])
    train_model(model, X_train, y_train, X_test, y_test)

    # Save the model to a file
    model.save('lstm_model.h5')

    # Save the scaler for future use in the FastAPI app
    joblib.dump(scaler_target, 'scaler_target.pkl')
    joblib.dump(scaler_features, 'scaler_features.pkl')

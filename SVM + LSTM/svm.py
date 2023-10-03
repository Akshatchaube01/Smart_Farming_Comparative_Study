import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data from the CSV file
def load_data(Crop_recommendation):
    data = pd.read_csv(Crop_recommendation)
    X = data[['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'PH', 'Rainfall']].values
    y = data['label'].values
    return X, y

# Preprocess data
def preprocess_data(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# Train SVM model
def train_svm(X_train, y_train):
    svm_model = SVC(kernel='linear', C=1.0)
    svm_model.fit(X_train, y_train)
    return svm_model

# Train LSTM model
def train_lstm(X_train, y_train):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
    lstm_model.add(Dense(1, activation='sigmoid'))
    
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)
    
    return lstm_model

# Main function
def main():
    # Load and preprocess data
    X, y = load_data('Crop_recommendation.csv')
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Train SVM model
    svm_model = train_svm(X_train, y_train)

    # Train LSTM model
    lstm_model = train_lstm(X_train, y_train)

    # Evaluate SVM model
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)

    # Evaluate LSTM model
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    lstm_pred = (lstm_model.predict(X_test_reshaped) > 0.5).astype(int)
    lstm_accuracy = accuracy_score(y_test, lstm_pred)

    print(f'SVM Accuracy: {svm_accuracy}')
    print(f'LSTM Accuracy: {lstm_accuracy}')

if _name_ == '_main_':
    main()

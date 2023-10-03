import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from pydeep.rbm import SupervisedDBNClassification

# Load data from CSV file
def load_data(Crop_recommendation):
    data = pd.read_csv(Crop_recommendation)
    X = data[['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'Ph', 'Rainfall']].values
    y = data['Crop_Label'].values  # Assuming 'Crop_Label' is the target variable
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

# Create and train a Deep Belief Network (DBN) model
def create_dbn_model(X_train, y_train):
    dbn = SupervisedDBNClassification(hidden_layers_structure=[64, 32], learning_rate_rbm=0.05,
                                      learning_rate=0.1, n_epochs_rbm=10, n_iter_backprop=100, batch_size=32,
                                      activation_function='relu')
    dbn.fit(X_train, y_train)
    return dbn

# Create and train an LSTM model
def create_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

# Combine DBN and LSTM models for prediction
def combined_model_prediction(dbn_model, lstm_model, X_test):
    # Use DBN to extract features
    dbn_features = dbn_model.transform(X_test)

    # Reshape DBN features for LSTM input
    dbn_features = dbn_features.reshape(dbn_features.shape[0], dbn_features.shape[1], 1)

    # Use LSTM for final prediction
    y_pred = (lstm_model.predict(dbn_features) > 0.5).astype(int)

    return y_pred

# Main function
def main():
    # Load data from CSV file
    csv_file = 'Crop_recommendation.csv'  # Replace with your CSV file path
    X, y = load_data(Crop_recommendation)

    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Create and train DBN and LSTM models
    dbn_model = create_dbn_model(X_train, y_train)
    lstm_model = create_lstm_model(X_train, y_train)

    # Make predictions using the combined model
    y_pred = combined_model_prediction(dbn_model, lstm_model, X_test)

    # Evaluate the combined model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Combined Model Accuracy: {accuracy:.2f}')

if _name_ == '_main_':
    main()

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import accuracy_score

def main():
    # Load data from CSV file (update the filename)
    data = pd.read_csv('Crop_recommendation.csv')

    # Extract features and target variable
    X = data[['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'Ph', 'Rainfall']]
    y = data['CropType']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create and train a Gaussian Naïve Bayes classifier
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(X_train, y_train)

    # Predict using the Naïve Bayes model
    y_pred_nb = naive_bayes_classifier.predict(X_test)

    # Calculate accuracy of Naïve Bayes
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f'Accuracy of Naïve Bayes Classifier: {accuracy_nb:.2f}')

    # Reshape data for LSTM (assuming a time series format)
    X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Create a sequential LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(1, 7)))  # 7 features
    lstm_model.add(Dense(1, activation='sigmoid'))  # Adjust the output layer based on your task

    # Compile the LSTM model
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the LSTM model
    lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=32, verbose=2)

    # Evaluate the LSTM model
    _, accuracy_lstm = lstm_model.evaluate(X_test_lstm, y_test)
    print(f'Accuracy of LSTM Model: {accuracy_lstm:.2f}')

if _name_ == "_main_":
    main()
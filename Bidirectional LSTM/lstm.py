import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout

# Load data from CSV file (replace 'your_data.csv' with your actual dataset)
def load_data(Crop_recommendation):
    data = pd.read_csv(Crop_recommendation)
    return data

# Preprocess data and train the model
def train_model(data):
    # Define the features (parameters)
    features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Humidity', 'Ph', 'Temperature', 'Rainfall']

    # Extract features and target variable (crop management decisions)
    X = data[features].values
    y = data['Crop_Management_Decision'].values  # Replace with your actual column name

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a Bidirectional LSTM model
    model = Sequential()
    model.add(Bidirectional(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.2%}')

    # Make predictions on the test data
    y_pred = (model.predict(X_test) > 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2%}')

    return model

# Main function
def main():
    # Load data from CSV file
    csv_file = 'Crop_recommendation.csv'  # Replace with your actual CSV file path
    data = load_data(Crop_recommendation)

    # Train the model and get the trained model
    trained_model = train_model(data)

    # You can use the trained_model for making predictions in your smart farming application.

if _name_ == '_main_':
    main()
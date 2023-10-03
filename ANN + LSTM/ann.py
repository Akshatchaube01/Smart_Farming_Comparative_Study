import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data from CSV
def load_and_preprocess_data(Crop_recommendation):
    data = pd.read_csv(Crop_recommendation)
    
    # Extract features and labels
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    
    # Encode labels if needed (e.g., if 'label' column contains string labels)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Build and train the ANN + LSTM model
def build_ann_lstm_model(input_shape):
    model = Sequential()
    
    # First, add an LSTM layer for sequential data
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    
    # Flatten the output of the LSTM layer
    model.add(Flatten())
    
    # Add a dense (fully connected) layer for ANN
    model.add(Dense(64, activation='relu'))
    
    # Output layer with sigmoid activation for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Main function
def main():
    csv_file = 'Crop_recommendation.csv'  # Replace with the path to your CSV file
    X_train, X_test, y_train, y_test = load_and_preprocess_data(Crop_recommendation)
    
    # Reshape the input data for LSTM
    input_shape = (X_train.shape[1], 1)
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Build the ANN + LSTM model
    model = build_ann_lstm_model(input_shape)
    
    # Train the model
    model.fit(X_train_reshaped, y_train, epochs=10, batch_size=32, verbose=2)
    
    # Evaluate the model on the testing data
    y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

if _name_ == '__main__':
    main()

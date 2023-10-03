import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
import matplotlib.pyplot as plt

def main():
    # Load data from CSV file 
    data = pd.read_csv('Crop_recommendation.csv')

    # Define the features (7 parameters)
    features = ['Nitrogen', 'Phosphorous', 'Potassium', 'Temperature', 'Humidity', 'Ph', 'Rainfall']

    # Extract features and target variable
    X = data[features].values
    y = data['Crop_Yield'].values  # Assuming 'Crop_Yield' is the target variable

    # Normalize features to the range [0, 1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape the input data for sequence models (LSTM/GRU)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Create a sequential model with both LSTM and GRU layers
    model = Sequential()
    model.add(LSTM(64, input_shape=(X_train.shape[1], 1), return_sequences=False, activation='relu'))
    model.add(GRU(64, activation='relu'))
    model.add(Dropout(0.2))  # Adding dropout for regularization
    model.add(Dense(1, activation='linear'))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the model
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=2)

    # Evaluate the model on the test data
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f'Test Loss: {loss:.4f}')

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Plot the training history (optional)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # You can now use y_pred for further analysis or visualization

if _name_ == "_main_":
    main()
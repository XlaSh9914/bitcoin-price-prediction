import streamlit as st
import numpy as np
import pandas as pd
import pickle
from keras.models import load_model

# Load the saved models and preprocessing tools
with open('linear_regression_model.pkl', 'rb') as lr_file:
    lr = pickle.load(lr_file)

with open('imputer.pkl', 'rb') as imp_file:
    imputer = pickle.load(imp_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Load the saved neural network model
model = load_model('neural_network_model.h5')

# Load dataset (only to get the column names)
df = pd.read_csv('bitcoin_dataset.csv').drop('Date', axis=1)

# Streamlit UI
st.title("Bitcoin Market Price Prediction")

# Calculate default values (mean of each feature)
X = df.drop('btc_market_price', axis=1)
default_values = {col: round(X[col].mean(), 2) for col in X.columns}

# Input fields for all features with default values
input_features = {}
for feature in X.columns:
    input_features[feature] = st.number_input(
        f"Enter {feature}:",
        value=default_values[feature]  # Use the mean as the default value
    )

# Predict Button Click Handler
if st.button('Predict Bitcoin Market Price'):
    # Collect input values from the user
    input_values = np.array([input_features[feature] for feature in X.columns]).reshape(1, -1)

    # Impute and scale the input values
    input_values_imputed = imputer.transform(input_values)
    input_values_scaled = scaler.transform(input_values_imputed)

    # 1. Linear Regression prediction
    lr_prediction = lr.predict(input_values_scaled)
    st.write(f"Linear Regression Prediction: ${lr_prediction[0]:.2f}")

    # 2. Neural Network prediction
    nn_prediction = model.predict(input_values_scaled)
    st.write(f"Neural Network Prediction: ${nn_prediction[0][0]:.2f}")
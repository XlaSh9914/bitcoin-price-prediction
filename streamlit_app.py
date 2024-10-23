import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('bitcoin_dataset.csv').drop('Date', axis=1)

# Feature and target split
X = df.drop('btc_market_price', axis=1)
Y = df['btc_market_price']

# Train-test split (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=92)

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Standard scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train_scaled, Y_train)

# Streamlit UI
st.title("Bitcoin Market Price Prediction")

# Calculate default values (mean of each feature)
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
    # Build and train Neural Network model on button click
    st.write("Training Neural Network, please wait...")

    # Build the model
    model = Sequential()
    model.add(Dense(25, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(5, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_absolute_error', optimizer='adam')

    # Train the model
    model.fit(X_train_scaled, Y_train, epochs=2000, verbose=2)

    st.write("Neural Network training completed!")

    # Collect input values from the user
    input_values = np.array([input_features[feature] for feature in X.columns]).reshape(1, -1)

    # Impute and scale the input values
    input_values_imputed = imputer.transform(input_values)
    input_values_scaled = scaler.transform(input_values_imputed)

    # Linear Regression prediction
    lr_prediction = lr.predict(input_values_scaled)
    st.write(f"Linear Regression Prediction: ${lr_prediction[0]:.2f}")

    # Neural Network prediction
    nn_prediction = model.predict(input_values_scaled)
    st.write(f"Neural Network Prediction: ${nn_prediction[0][0]:.2f}")
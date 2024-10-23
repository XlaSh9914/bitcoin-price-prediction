import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import pickle

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

# 1. Train Linear Regression model
lr = LinearRegression()
lr.fit(X_train_scaled, Y_train)

# Save the Linear Regression model to a file
with open('linear_regression_model.pkl', 'wb') as lr_file:
    pickle.dump(lr, lr_file)
print("Linear Regression model saved as 'linear_regression_model.pkl'")

# 2. Build and Train Neural Network Model
model = Sequential()
model.add(Dense(25, input_dim=X.shape[1], activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_absolute_error', optimizer='adam')

# Train the Neural Network
model.fit(X_train_scaled, Y_train, epochs=2000, verbose=2)

# Save the trained Neural Network model
model.save('neural_network_model.h5')
print("Neural Network model saved as 'neural_network_model.h5'")

# Save the Imputer and Scaler used for preprocessing
with open('imputer.pkl', 'wb') as imp_file:
    pickle.dump(imputer, imp_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

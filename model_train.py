import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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

# Make predictions for the training set
Y_train_pred = lr.predict(X_train_scaled)

# Create a new figure for the plot
plt.figure(figsize=(10, 6))

# Scatter plot of the actual target values vs. predicted values
plt.scatter(Y_train, Y_train_pred, color='blue', label='Actual vs Predicted')

# Plot the regression line (fitted line)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2, label='Ideal Fit')  # Identity line

# Scatter plot for the fitted regression line
sorted_indices = np.argsort(Y_train)  # Sort the actual values for plotting the line
plt.plot(Y_train.iloc[sorted_indices], Y_train_pred[sorted_indices], color='orange', label='Regression Line')

# Title and labels
plt.title('Linear Regression: Actual vs Predicted (Training Data)')
plt.xlabel('Actual Bitcoin Market Price')
plt.ylabel('Predicted Bitcoin Market Price')
plt.legend()
plt.grid()

# Save the plot as a PNG file
plt.savefig('linear_regression_training_data_plot.png', dpi=300)
plt.close()  # Close the plot to free up memory

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

# Save the model architecture as a PNG image with enhanced options
plot_model(model, to_file='neural_network_architecture.png', 
           show_shapes=True,        # Show shape of layers
           show_layer_names=True,   # Show layer names
           rankdir='TB',            # Direction of the graph
           dpi=300)                 # Increase the resolution

print("Neural network architecture saved as 'neural_network_architecture.png'.")

# Save the Imputer and Scaler used for preprocessing
with open('imputer.pkl', 'wb') as imp_file:
    pickle.dump(imputer, imp_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

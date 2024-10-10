import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load preprocessed data
df = pd.read_csv('processed_weather_data.csv', index_col='time', parse_dates=True)

# Select the target variable
data = df['tavg'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Prepare the data for LSTM
def create_dataset(dataset, look_back=1):
    X, y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

look_back = 10  # Number of previous time steps to consider
X, y = create_dataset(scaled_data, look_back)

# Split into train and test sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape input to be [samples, time steps, features]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Evaluate the model using MSE
mse = mean_squared_error(y_test, predictions)
print(f"LSTM Mean Squared Error: {mse}")

# Save LSTM predictions to CSV
lstm_predictions_df = pd.DataFrame({'time': df.index[train_size + look_back + 1:], 'prediction': predictions.flatten()})
lstm_predictions_df.to_csv('lstm_predictions.csv', index=False)


# Plot actual vs predicted
plt.plot(df.index[train_size + look_back + 1:], scaler.inverse_transform(scaled_data[train_size + look_back + 1:]), label='Actual')
plt.plot(df.index[train_size + look_back + 1:], predictions, label='Predicted', color='red')
plt.title('LSTM Forecast vs Actual')
plt.legend()
plt.show()

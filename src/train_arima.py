import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load preprocessed data
df = pd.read_csv('processed_weather_data.csv', index_col='time', parse_dates=True)

# Select the target variable for ARIMA (e.g., 'tavg')
data = df['tavg']

# Split into training and test sets (80% train, 20% test)
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Train ARIMA model (adjust p, d, q parameters based on your data)
model = ARIMA(train, order=(5, 1, 0))  # ARIMA(p, d, q), adjust as needed
model_fit = model.fit()

# Make predictions
predictions = model_fit.forecast(steps=len(test))

# Evaluate the model using Mean Squared Error (MSE)
mse = mean_squared_error(test, predictions)
print(f"ARIMA Mean Squared Error: {mse}")

# Save ARIMA predictions to CSV
predictions_df = pd.DataFrame({'time': test.index, 'prediction': predictions})
predictions_df.to_csv('arima_predictions.csv', index=False)

# Plot actual vs predicted
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, label='Predicted', color='red')
plt.title('ARIMA Forecast vs Actual')
plt.legend()
plt.show()

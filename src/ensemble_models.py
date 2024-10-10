import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load the predictions from the three models
arima_predictions = pd.read_csv('arima_predictions.csv', index_col='time')
prophet_predictions = pd.read_csv('prophet_predictions.csv', index_col='time')
lstm_predictions = pd.read_csv('lstm_predictions.csv', index_col='time')

# Load the actual test data
actual_test_data = pd.read_csv('actual_test_data.csv', index_col='time')

# Find common dates across all predictions and actual test data
common_dates = actual_test_data.index.intersection(arima_predictions.index)
common_dates = common_dates.intersection(prophet_predictions.index)
common_dates = common_dates.intersection(lstm_predictions.index)

# Filter predictions and actual data to only include the common dates
actual_test_data = actual_test_data.loc[common_dates]
arima_predictions = arima_predictions.loc[common_dates]
prophet_predictions = prophet_predictions.loc[common_dates]
lstm_predictions = lstm_predictions.loc[common_dates]

# Combine predictions using weighted averaging (adjust weights as needed)
weights = [0.4, 0.3, 0.3]  # Example weights for ARIMA, Prophet, LSTM
ensemble_predictions = (weights[0] * arima_predictions['prediction'] +
                        weights[1] * prophet_predictions['yhat'] +
                        weights[2] * lstm_predictions['prediction'])

# Save ensemble predictions to a CSV file in the project root directory (outside src)
ensemble_df = pd.DataFrame({'time': common_dates, 'prediction': ensemble_predictions})
ensemble_df.to_csv('./ensemble_predictions.csv', index=False)  # Adjusted file path
print("Ensemble predictions saved to '../ensemble_predictions.csv'.")

# Evaluate the ensemble model
mse = mean_squared_error(actual_test_data['tavg'], ensemble_predictions)
print(f"Ensemble Model Mean Squared Error: {mse}")

# Plot the actual vs predicted
plt.plot(actual_test_data.index, actual_test_data['tavg'], label='Actual')
plt.plot(actual_test_data.index, ensemble_predictions, label='Ensemble Predicted', color='red')
plt.title('Ensemble Model Forecast vs Actual')
plt.legend()
plt.show()

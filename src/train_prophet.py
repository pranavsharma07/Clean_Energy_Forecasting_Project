import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Load preprocessed data
df = pd.read_csv('processed_weather_data.csv', index_col='time', parse_dates=True)

# Prepare the data for Prophet (Prophet requires 'ds' and 'y' columns)
prophet_df = df[['tavg']].reset_index()
prophet_df.columns = ['ds', 'y']

# Split into training and test sets
train_size = int(len(prophet_df) * 0.8)
train, test = prophet_df[:train_size], prophet_df[train_size:]

# Initialize and train the Prophet model
model = Prophet()
model.fit(train)

# Make future predictions
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

# Evaluate the model using MSE
predictions = forecast[['ds', 'yhat']].set_index('ds').loc[test['ds']]
mse = mean_squared_error(test['y'], predictions['yhat'])
print(f"Prophet Mean Squared Error: {mse}")

# Save Prophet predictions to CSV
prophet_predictions_df = forecast[['ds', 'yhat']].rename(columns={'ds': 'time'})
prophet_predictions_df.to_csv('prophet_predictions.csv', index=False)

# Plot the forecast
model.plot(forecast)
plt.title('Prophet Forecast')
plt.show()

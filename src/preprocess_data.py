import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load your historical weather data
df = pd.read_csv('historical_weather_data_backup.csv', index_col='time', parse_dates=True)

# Drop columns with too many missing values
df.drop(columns=['snow', 'tsun'], inplace=True)

# Inspect the original dataset after dropping these columns
print(f"Dataset shape after dropping 'snow' and 'tsun': {df.shape}")
print(f"Missing values in each column after dropping 'snow' and 'tsun':\n{df.isnull().sum()}")

# Handle remaining missing values (forward and backward fill)
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Feature engineering: Create lag features and moving averages
df['tavg_lag1'] = df['tavg'].shift(1)  # Lag by 1 day
df['tavg_ma3'] = df['tavg'].rolling(window=3).mean()  # 3-day moving average

# Drop rows with NaN values created by lagging/rolling
df.dropna(inplace=True)

# Check if the DataFrame is empty after dropping NaNs
if df.empty:
    print("No data available after feature engineering. Please check the input data.")
else:
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df[['tavg', 'tavg_lag1', 'tavg_ma3']])

    # Convert back to a DataFrame
    processed_df = pd.DataFrame(scaled_features, index=df.index, columns=['tavg', 'tavg_lag1', 'tavg_ma3'])

    # Save processed data to CSV for model training
    processed_df.to_csv('processed_weather_data.csv')
    print("Data preprocessing complete. Processed data saved to 'processed_weather_data.csv'.")

    # --- NEW CODE TO SAVE ACTUAL TEST DATA ---
    # Split the data into training and test sets (80% train, 20% test)
    train_size = int(len(processed_df) * 0.8)
    test_data = processed_df[train_size:]  # Test data

    # Save the actual test data to CSV for comparison with predictions
    actual_test_data = pd.DataFrame({'time': test_data.index, 'tavg': test_data['tavg']})
    actual_test_data.to_csv('actual_test_data.csv', index=False)

    print("Actual test data saved to 'actual_test_data.csv'.")

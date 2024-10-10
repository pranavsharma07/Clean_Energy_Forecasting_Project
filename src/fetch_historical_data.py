from meteostat import Point, Daily
import pandas as pd
from datetime import datetime

# Define the location (latitude, longitude, and altitude)
# Example: London (51.51, -0.13, 35m altitude)
location = Point(51.51, -0.13, 35)

# Define the start and end dates (2 years of data)
start = datetime(2022, 9, 1)  # 2 years ago from today (customize if needed)
end = datetime(2024, 9, 1)    # Today's date

# Fetch daily weather data for this period
data = Daily(location, start, end)
data = data.fetch()

# Convert to a Pandas DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the data
print(df.head())

# Optional: Save the DataFrame to CSV or SQLite
df.to_csv('historical_weather_data.csv', index=True)

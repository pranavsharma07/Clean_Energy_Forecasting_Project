import requests
import pandas as pd
import sqlite3

# Your OpenWeatherMap API key
API_KEY = '5ef1f079bab6efddc3455f17b9e20538'

# Define the URL for the API (example: OpenWeather API)
city = 'London'
url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric'

# Make the request to the API
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    data = response.json()

    # Convert relevant data into a Pandas DataFrame
    weather_data = {
        'city': data['name'],
        'temperature': data['main']['temp'],
        'humidity': data['main']['humidity'],
        'weather': data['weather'][0]['description']
    }

    df = pd.DataFrame([weather_data])

    # Print the DataFrame
    print(df)

    # Step 1: Connect to SQLite Database (or create it if it doesn't exist)
    conn = sqlite3.connect('energy_data.db')

    # Step 2: Create a table for weather data (if it doesn't exist)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS weather_data (
            city TEXT,
            temperature REAL,
            humidity INTEGER,
            weather TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Step 3: Insert the DataFrame into the SQLite table
    df.to_sql('weather_data', conn, if_exists='append', index=False)

    # Step 4: Commit and close the connection
    conn.commit()
    conn.close()

    print("Data saved to SQLite database successfully.")

else:
    print(f"Error: {response.status_code}")

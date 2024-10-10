import sqlite3
import pandas as pd

# Load the historical weather data (assuming it was fetched earlier and saved as a DataFrame)
df = pd.read_csv('historical_weather_data.csv')

# 1. Save Data to a CSV File
df.to_csv('historical_weather_data_backup.csv', index=True)
print("Data saved to historical_weather_data_backup.csv.")

# 2. Save Data to SQLite
def save_to_sqlite(dataframe, db_name='energy_data.db'):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    
    # Save the DataFrame to the SQLite table (append mode)
    dataframe.to_sql('historical_weather_data', conn, if_exists='append', index=True)
    
    # Commit and close the connection
    conn.commit()
    conn.close()
    print("Data saved to SQLite database.")

# Call the function to save to SQLite
save_to_sqlite(df)

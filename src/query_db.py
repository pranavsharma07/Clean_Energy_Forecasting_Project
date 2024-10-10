import sqlite3
import pandas as pd

# Connect to the SQLite database
conn = sqlite3.connect('energy_data.db')

# Query the data
df = pd.read_sql_query("SELECT * FROM weather_data", conn)

# Check if the DataFrame is empty
if df.empty:
    print("No data found in the weather_data table.")
else:
    print(df)

# Close the connection
conn.close()

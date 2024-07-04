"""
Used to convert the CSV file to a SQLite database table. This script is used in preparation for 
the API to query the SQLite database.
"""

import sqlite3
import pandas as pd
from discord_bot.parameters import PRODUCT_DESCRIPTIONS_CSV, SQLITE_DB_FILE, SQL_TABLE_NAME

try:
    # Read the CSV file into a DataFrame
    # Attempt using UTF-8 encoding first
    try:
        df = pd.read_csv(PRODUCT_DESCRIPTIONS_CSV)
    except UnicodeDecodeError:
        # If UTF-8 fails, fall back to ISO-8859-1
        df = pd.read_csv(PRODUCT_DESCRIPTIONS_CSV, encoding='ISO-8859-1')
    
    # Connect to the SQLite database
    conn = sqlite3.connect(SQLITE_DB_FILE)

    # Write the DataFrame to a SQLite table
    df.to_sql(SQL_TABLE_NAME, conn, if_exists='replace', index=False)

    # Commit the changes
    conn.commit()

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    # Close the connection
    conn.close()
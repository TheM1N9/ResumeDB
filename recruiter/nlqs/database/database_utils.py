import os
import pandas as pd
from typing import Optional
import sqlite3
from dotenv import load_dotenv
load_dotenv()

LOGGER_FILE = os.getenv("LOGGER_FILE")
SQLITE_DB_FILE = os.getenv("SQLITE_DB_FILE")
SQL_TABLE_NAME = os.getenv("SQL_TABLE_NAME")

def fetch_data_from_sqlite(db_file: str= SQLITE_DB_FILE, table_name: str= SQL_TABLE_NAME) -> Optional[pd.DataFrame]:
    try:
        conn = sqlite3.connect(db_file)
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()
    except sqlite3.Error as e:
        print(f"Error fetching data: {e}")
        return None  # Return an empty DataFrame on error
     
    return df
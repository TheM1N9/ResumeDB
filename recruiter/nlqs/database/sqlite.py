import re
import sqlite3
import logging
from typing import Optional, Tuple
# from discord_bot.parameters import LOGGER_FILE, SQLITE_DB_FILE
from recruiter.nlqs import query
from recruiter.nlqs.database.driver import AbstractDriver
import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

LOGGER_FILE = os.getenv("LOGGER_FILE")
SQLITE_DB_FILE = os.getenv("SQLITE_DB_FILE")
SQL_TABLE_NAME = os.getenv("SQL_TABLE_NAME")

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)

# Create a file handler to save logs
file_handler = logging.FileHandler(LOGGER_FILE)

# Create a formatter to format the log messages
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Add the formatter to the file handler
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)

class SQLiteDriver(AbstractDriver):

    def __init__(self, config):
        pass

    def connect(self):
        pass

    def disconnect(self):
        pass

    def execute_query(self, query):
        pass
    
    def retrieve_descriptions_and_types_from_db(self, db_file):
        pass

    def validate_query(self, query, db_file):
        pass

# def fetch_data_from_sqlite(db_file: str= SQLITE_DB_FILE, table_name: str= SQL_TABLE_NAME) -> Optional[pd.DataFrame]:
#     try:
#         conn = sqlite3.connect(db_file)
#         query = f"SELECT * FROM {table_name}"
#         df = pd.read_sql_query(query, conn)
#         conn.close()
#     except sqlite3.Error as e:
#         print(f"Error fetching data: {e}")
#         return None  # Return an empty DataFrame on error
     
#     return df


def retrieve_descriptions_and_types_from_db(db_file: str= SQLITE_DB_FILE) -> Tuple:
    """ Retrieves descriptions and types from the SQLite database.

    Args:
        db_file (SQLite database, optional): SQLite database to store all the tables. Defaults to SQLITE_DB_FILE.

    Returns:
        tuple: Return descriptions, numerical_columns, categorial_columns
    """
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # Retrieve descriptions
    c.execute('SELECT column_name, description FROM column_descriptions')
    description_rows = c.fetchall()
    descriptions = {row[0]: row[1] for row in description_rows}

    # Retrieve column types
    c.execute('SELECT column_name, column_type FROM column_types')
    type_rows = c.fetchall()
    numerical_columns = [row[0] for row in type_rows if row[1] == 'numerical']
    categorical_columns = [row[0] for row in type_rows if row[1] == 'categorical']

    conn.close()
    return descriptions, numerical_columns, categorical_columns

def validate_query(query: str, db_file: str = SQLITE_DB_FILE) -> bool:
    """ Validates the generated SQL query against the database schema and returns True if valid, False otherwise.

    Args:
        query (str): sql query to be validated.
        db_file (str, optional): sqlite database file to be used. Defaults to SQLITE_DB_FILE.

    Returns:
        bool: True if query is valid, False otherwise.
    """
    # Check for basic syntax errors
    if not query.strip() or not query.lower().startswith("select"):
        return False

    # Connect to the database
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return False

    # Extract table name and columns from the query
    try:
        match = re.search(r"FROM\s+(.+?)\s+", query, re.IGNORECASE)
        if match is None:
            return False
        table_name = match.group(1).strip()
        columns = re.findall(r"SELECT\s+(.+?)\s+FROM", query, re.IGNORECASE)[0].strip().split(",")
    except AttributeError:
        print("Error extracting table name or columns from query.")
        return False

    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    if not cursor.fetchone():
        print(f"Table '{table_name}' does not exist in the database.")
        return False

    # Check if columns exist in the table
    cursor.execute("PRAGMA table_info({})".format(table_name))
    table_columns = [column[1] for column in cursor.fetchall()]
    for column in columns:
        column = column.strip()
        if column not in table_columns and column != '*':
            print(f"Column '{column}' does not exist in table '{table_name}'.")
            return False
        
    # If all checks pass, the query is considered valid
    return True

def execute_query(query:str) -> str:
    """ Executes the SQL query and returns the result.

    Args:
        query (str): the SQL query.

    Returns:
        str: the result of the query.
    """
    try:
        #SQLite connection
        conn = sqlite3.connect(SQLITE_DB_FILE)
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchall()
        conn.close()
        result = str(result)
        logger.info(f"Query Result: {result}")

        return result if result else "No results found."
    except sqlite3.Error as e:
        error_message = f"Error executing SQL query: {e}"
        logger.error(f"Error executing SQL query: {e}")
        return error_message
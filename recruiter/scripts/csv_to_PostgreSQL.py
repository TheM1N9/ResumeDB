"""
Used to convert the CSV file to a PostgreSQL database table. This script is used in preparation for 
the API to query the PostgreSQL database.
"""

import pandas as pd
from sqlalchemy import create_engine

# Load the Excel file
csv_file = 'product_descriptions.csv'

# Read the Excel file into a pandas DataFrame
df = pd.read_csv(csv_file)

# Database connection parameters
db_user = 'postgres'
db_password = '1234'
db_host = 'localhost'
db_port = '5433'  # default PostgreSQL port is 5432
db_name = 'postgres'

# Create the connection string
connection_string = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

# Create the database engine
engine = create_engine(connection_string)

# Define the table name
table_name = 'new_dataset'

# Insert data into PostgreSQL
df.to_sql(table_name, engine, if_exists='replace', index=False)

from pathlib import Path
from discord_bot.parameters import SQL_TABLE_NAME, SQLITE_DB_FILE
from nlqs.column_introspection import fetch_data_from_sqlite, generate_sample_data, get_column_descriptions, store_descriptions_in_db


df = fetch_data_from_sqlite(Path(SQLITE_DB_FILE),SQL_TABLE_NAME)


if df is None:
    print("Error fetching data from SQLite.")
    raise Exception("Error fetching data from SQLite.")

# Generate sample data for each column
sample_data_list = generate_sample_data(df)

# Get column descriptions from OpenAI API
column_descriptions = get_column_descriptions(sample_data_list, input_text="Please provide a detailed description of each column in the given dataset.")

# Identify numerical and categorical columns
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# Store descriptions and column types in the database
store_descriptions_in_db(column_descriptions, numerical_columns, categorical_columns, SQLITE_DB_FILE)

print("Column descriptions and column types stored in the database.")
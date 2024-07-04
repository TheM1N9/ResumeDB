from pathlib import Path
from re import S
import sqlite3
from typing import Dict, List, Optional
import pandas as pd
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
# from discord_bot.parameters import OPENAI_API_KEY, SQLITE_DB_FILE, SQL_TABLE_NAME

# Fetch data from SQLite
def fetch_data_from_sqlite(db_file: Path, table_name: str) -> Optional[pd.DataFrame]:
    """Fetch data from a SQLite database table.

    Args:
        db_file (Path): Path to the SQLite database file.
        table_name (str): Name of the table to fetch data from.

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the data from the table, or None if an error occurred.
    """
    try:
        conn = sqlite3.connect(db_file)
        query = f"SELECT * FROM {table_name}"
        df = pd.read_sql_query(query, conn)
        conn.close()
    except sqlite3.Error as e:
        print(f"Error fetching data: {e}")
        return None  # Return an empty DataFrame on error
     
    return df

def generate_sample_data(dataframe: pd.DataFrame) -> List[str]:
    """Generate sample data for each column in the DataFrame.

    Args:
        dataframe (pd.DataFrame): The DataFrame containing the data.

    Returns:
        List[str]: A list of strings containing the column name, sample data, and data type.
    """
    sample_data_list = []
    for column in dataframe.columns:
        col_data = dataframe[column]
        col_type = col_data.dtype
        sample_data = col_data.dropna().sample(min(5, len(col_data))).tolist()
        sample_data_str = ', '.join(map(str, sample_data))
        sample_data_list.append(f"{column}: {sample_data_str}, {col_type}")
    return sample_data_list

def get_column_descriptions(sample_data: List[str], input_text: str) -> Dict:
    """Generate descriptions for each column in the dataset using OpenAI

    Args:
        sample_data (str): Sample data for each of the columns in the dataset formatted as ???.
        input_text (str): The input text to prompt the user for column descriptions.

    Returns:
        Dict: _description_
    """    
    # Initialize an empty dictionary to store column descriptions
    descriptions = {}

    for item in sample_data:
        column_info, col_type = item.rsplit(", ", 1)
        column, sample_data = column_info.split(": ", 1)
        print(column, col_type, sample_data)

        # Prepare the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"""
                The following is a description of a dataset column:
                Column Name: {column}
                Data Type: {col_type}
                
                Please provide a detailed description of this column, including its potential meaning, use, and importance in a dataset. Use sample data to identify the column's meaning.
                
                Sample Data: {sample_data}
                
                Use the following format:

                For example:
                   "Product": "This column contains the name of the product. It is a text field and can be used for exact or partial matches.",
                   "Category": "This column contains the category of the product. It is a text field and can be used for exact or partial matches.",
                    "MedicalBenefits": "This column contains the medical benefits of the product. It is a text field and can be used for exact or partial matches.",
                    "CustomerRating": "This column contains the customer rating of the product. It is a numerical field and can be used for exact matches or range comparisons.",
                    "PurchaseFrequency": "This column contains the frequency of product purchase. It is a text field and can be used for exact or partial matches.",
                    "description": "This column contains the description of the product. It is a text field and can be used for exact or partial matches."
                """),
                ("user", "{user_input}")
            ]
        )

        llm = ChatOpenAI(
            model="gpt-4",
            api_key=OPENAI_API_KEY,
            temperature=0.0,
            verbose=True,
        )

        chain = LLMChain(prompt=prompt, llm=llm)
        
        # Generate the description
        response = chain.run(input_text)
        
        # Extract the description from the response
        description = response.strip()
        
        # Add the description to the dictionary
        descriptions[column] = description
        
    # Return the dictionary of column descriptions
    return descriptions

def store_descriptions_in_db(descriptions: Dict[str, str], numerical_columns: List[str], categorical_columns: List[str], db_file: Path) -> None:
    """Store column descriptions and types in the SQLite database.

    Args:
        descriptions (Dict[str, str]): Dictionary of column descriptions.
        numerical_columns (List[str]): List of numerical columns.
        categorical_columns (List[str]): List of categorical columns.
        db_file (Path): Path to the SQLite database file.
    """
    conn = sqlite3.connect(db_file)
    c = conn.cursor()

    # Create table for column descriptions
    c.execute('''
        CREATE TABLE IF NOT EXISTS column_descriptions (
            column_name TEXT PRIMARY KEY,
            description TEXT
        )
    ''')

    for column, description in descriptions.items():
        c.execute('''
            INSERT OR REPLACE INTO column_descriptions (column_name, description)
            VALUES (?, ?)
        ''', (column, description))

    # Create table for numerical and categorical columns
    c.execute('''
        CREATE TABLE IF NOT EXISTS column_types (
            column_name TEXT PRIMARY KEY,
            column_type TEXT
        )
    ''')

    for column in numerical_columns:
        c.execute('''
            INSERT OR REPLACE INTO column_types (column_name, column_type)
            VALUES (?, ?)
        ''', (column, 'numerical'))

    for column in categorical_columns:
        c.execute('''
            INSERT OR REPLACE INTO column_types (column_name, column_type)
            VALUES (?, ?)
        ''', (column, 'categorical'))

    conn.commit()
    conn.close()

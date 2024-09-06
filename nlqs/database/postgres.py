import logging
import re
import psycopg2
from psycopg2 import sql
from typing import Dict, List, Optional, Tuple

import pandas as pd

from nlqs.database.abstract_driver import AbstractDriver

from dataclasses import dataclass

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)

@dataclass
class PostgresConnectionConfig:
    host: str
    port: int
    user: str
    password: str
    database_name: str
    dataset_table_name: str


class PostgresDriver(AbstractDriver):
    def __init__(self, pg_config: PostgresConnectionConfig):
        """
        
        Example:
        postgres_config = PostgresConnectionConfig(
            host="localhost", 
            port=5432, 
            user="postgres",  
            password="password",
            database_name="aegion",  
            dataset_table_name="new_dataset"  
        )

        """
        self.db_config = pg_config
        self._db_connection = None
        self.cursor = None

    def connect(self):
        try:
            self._db_connection = psycopg2.connect(
                dbname=self.db_config.database_name, 
                user=self.db_config.user, 
                password=self.db_config.password, 
                host=self.db_config.host, 
                port=self.db_config.port
            )            
            self.cursor = self._db_connection.cursor()
            logger.info("Connected to PostgreSQL database.")
        except psycopg2.Error as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def disconnect(self):
        if self._db_connection:
            self._db_connection.close()
            logger.info("Disconnected from PostgreSQL database.")

    def execute_query(self, query: str) -> List[str]:
        """Executes the SQL query and returns the result.

        Args:
            query (str): the SQL query.

        Returns:
            List[str]: the result of the query.
        """
        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")
        try:
            print(f"Executing query: {query}")
            self.cursor.execute(query)
            result = self.cursor.fetchall()
            self._db_connection.commit()
            print(f"result: {result}")
            print("--------------------")
            logger.info(f"Query executed successfully: {result}")
            return result if result else []
        except psycopg2.Error as e:
            error_message = f"Error executing SQL query: {e}"
            logger.error(error_message)
            raise e
            # return error_message

    def retrieve_descriptions_and_types_from_db(self) -> Tuple[Dict[str, str], List[str], List[str]]:
        """Retrieves descriptions and types from the PostgreSQL database.

        Args:
            db_file (PostgreSQL database, optional): PostgreSQL database to store all the tables. Defaults to POSTGRES_DB_CONFIG.

        Returns:
            Tuple[List[str], List[str], List[str]]: Return descriptions, numerical_columns, categorial_columns
        """
        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")

        try:
            # Retrieve descriptions
            self.cursor.execute("SELECT column_name, description FROM column_descriptions")
            description_rows = self.cursor.fetchall()
            descriptions = {row[0]: row[1] for row in description_rows}

            # Retrieve column types
            self.cursor.execute("SELECT column_name, column_type FROM column_types")
            type_rows = self.cursor.fetchall()
            numerical_columns = [row[0] for row in type_rows if row[1] == "numerical"]
            categorical_columns = [row[0] for row in type_rows if row[1] == "categorical"]

            return descriptions, numerical_columns, categorical_columns
        except psycopg2.Error as e:
            logger.error(f"Error retrieving descriptions and types: {e}")
            return {}, [], []
        
    def database_columns(self) -> list[str]:
        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")
        try:
            self.cursor.execute("SELECT column_name FROM column_types")
            columns = self.cursor.fetchall()
            return columns
        except psycopg2.Error as e:
            logger.error(f"Error retrieving columns: {e}")
            return []

    def validate_query(self, query: str) -> bool:
        """Validates the generated SQL query against the database schema and returns True if valid, False otherwise.

        Args:
            query (str): sql query to be validated.

        Returns:
            bool: True if query is valid, False otherwise.
        """
        if not query.strip() or not query.lower().startswith("select"):
            return False
        
        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")
        
        try:
            match = re.search(r"FROM\s+(\w+)", query, re.IGNORECASE)
            if not match:
                return False
            table_name = match.group(1).strip()

            column_match = re.search(r"SELECT\s+(.+?)\s+FROM", query, re.IGNORECASE)
            if not column_match:
                return False
            columns = column_match.group(1).strip().split(",")

            self.cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name=%s", (table_name,))
            if not self.cursor.fetchone():
                return False

            self.cursor.execute(sql.SQL("SELECT column_name FROM information_schema.columns WHERE table_name = %s"), [table_name])
            table_columns = [row[0] for row in self.cursor.fetchall()]
            for column in columns:
                column = column.strip()
                if column not in table_columns and column != "*":
                    return False

            return True
        except psycopg2.Error as e:
            logger.error(f"Error validating query: {e}")
            return False


    def fetch_data_from_database(self, table_name: str) -> pd.DataFrame:
        """Fetch data from a PostgreSQL database table.

        Args:
            db_file (Path): Path to the PostgreSQL database file.
            table_name (str): Name of the table to fetch data from.

        Returns:get_primary_key
            pd.DataFrame: A DataFrame containing the data from the table, or None if an error occurred.
        """
        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")
        try:
            conn = self._db_connection
            query = f"SELECT * FROM {table_name}"
            df = pd.read_sql_query(query, conn) # type: ignore
            conn.close()
        except psycopg2.Error as e:
            logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()  # Return an empty DataFrame on error
            
        return df
    
    def get_primary_key(self, table_name: str) -> str:
        """
        Retrieves the primary key column name from a SQLite table.

        Args:
            table_name (str): The name of the table to check.

        Returns:
            primary key (str): The name of the primary key column, 
                           or None if no primary key is found.
        """
        if self.cursor is None or self._db_connection is None:
            raise ValueError("Database connection not established.")

        try:
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            table_info = self.cursor.fetchall()

            for row in table_info:
                if row[5] == 1:  # Check for primary key indicator
                    return row[1]  # Return the column name

            # No primary key found
            raise ValueError("No primary key found in the database.")

        except psycopg2.Error as e:
            print(f"Error getting primary key: {e}")
            raise e

    @property
    def db_connection(self):
        if self._db_connection is None:
            raise ValueError("Database connection not established.")
        return self._db_connection
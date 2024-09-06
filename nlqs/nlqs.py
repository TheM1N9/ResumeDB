import logging
from nlqs.database.postgres import PostgresDriver, PostgresConnectionConfig
from nlqs.database.sqlite import SQLiteDriver, SQLiteConnectionConfig
import re
from typing import Dict, List, Tuple, Union
from nlqs.description_generator import (
    generate_column_description,
    get_chroma_collection,
)
from nlqs.query import (
    generate_quantitaive_serach_query,
    qualitative_search,
    summarize,
)
from dataclasses import dataclass
from pathlib import Path
from langchain_openai import ChatOpenAI
from pydantic.v1 import SecretStr
from nlqs.parameters import OPENAI_API_KEY
from nlqs.parameters import LOGGER_FILE

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


@dataclass
class ChromaDBConfig:
    collection_name: str
    persist_path: Path
    is_local: bool = True


class NLQS:

    def __init__(
        self,
        connection_config: Union[SQLiteConnectionConfig, PostgresConnectionConfig],
        chroma_config: ChromaDBConfig,
        chroma_client,
    ) -> None:
        # TODO - Figure out what the constructor parameters are
        if isinstance(connection_config, SQLiteConnectionConfig):
            self.connection_driver = SQLiteDriver(connection_config)
        elif isinstance(connection_config, PostgresConnectionConfig):
            self.connection_driver = PostgresDriver(connection_config)

        # Initialize the connection to the database
        self.connection_driver.connect()

        # Create the llm object
        # Initializes the ChatOpenAI LLM model
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4-turbo",
            api_key=SecretStr(OPENAI_API_KEY),
            max_tokens=1000,
        )

        self.chroma_config = chroma_config
        self.chroma_client = chroma_client

        # TODO - Figure out if we need to create introspection table, and create
        pass

    def _create_introspection_table(self):
        driver = self.connection_driver

        # Step 1
        column_descriptions, numerical_columns, categorical_columns = (
            driver.retrieve_descriptions_and_types_from_db()
        )

        if column_descriptions == {}:
            # Step 2
            generate_column_description(
                df=self.connection_driver.fetch_data_from_database(
                    table_name=self.connection_driver.db_config.dataset_table_name
                ),
                db_driver=self.connection_driver,
            )
            column_descriptions, numerical_columns, categorical_columns = (
                driver.retrieve_descriptions_and_types_from_db()
            )

        return column_descriptions, numerical_columns, categorical_columns

    # Step 4
    def execute_nlqs_workflow(
        self, user_input: str, chat_history: List[Tuple[str, str]]
    ) -> str:
        """This function is where the whole interaction happens.
        It takes the user input and chat history as input and returns the response if the user's intent is either phatic_communication, profanity or sql_injection.
        Else it returns the query result or search similarity result and the updated chat history.

        Args:
            user_input (str): The user's input.
            chat_history (list[(str, str)]): The chat history.

        Returns:
            Tuple[str,List[Tuple[str, str]]]: The response and the updated chat history.
        """

        # Overview
        # Step 1 - retrieve descriptions and types from db. check if its empty. if not return the data.
        # Step 2 - else if the retrived data was empty then generate new columns descriptions.
        # Step 3 - next get the chroma collection
        # Step 4 - pass all the retrieved data to the main_workflow method
        # Step 5 - check if the user input is empty if true retun none
        # Step 6 - Else remove the paranthesis from the user input.
        # Step 7 - generate a summary for the user input the required format.
        # Step 8 - check if the summary is empty. if true retry the generation of the summary, you can do this until five times
        # (the above step is because we were getting errors while converting the generted summary to the json format.)
        # Step 9 - generate an sql query.
        # Step 10 - validate the generated query.
        # Step 11 - check if the query result is empty. if true then do a similarity search and retrieve the relevent info and return it.
        # Step 12 - else return the query result.

        # Step 0 - Create the pre-requisite objects

        # Database Connection
        driver = self.connection_driver

        column_descriptions, numerical_columns, categorical_columns = (
            self._create_introspection_table()
        )

        primary_key = driver.get_primary_key(driver.db_config.dataset_table_name)

        # Chroma Collection
        chroma_collections = get_chroma_collection(
            collection_name=self.chroma_config.collection_name,
            client=self.chroma_client,
            db_driver=driver,
            primary_key=primary_key,
        )

        # Step 5
        if not user_input.strip():
            response = ""

        # Step 6
        user_input = re.sub(r"{|}", "", user_input)

        # Step 7
        summarized_input = summarize(
            user_input=user_input,
            chat_history=chat_history,
            column_descriptions_dictionary=column_descriptions,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
            llm=self.llm,
        )

        count = 0
        print(f"summarized_input: {summarized_input}")
        while not summarized_input.summary and count < 5:
            summarized_input = summarize(
                user_input=user_input,
                chat_history=chat_history,
                column_descriptions_dictionary=column_descriptions,
                numerical_columns=numerical_columns,
                categorical_columns=categorical_columns,
                llm=self.llm,
            )
            count += 1
            if count == 5:
                response = "Summarization failed. Please try again."
                break

        intent = summarized_input.user_intent

        logger.info("--------------------------")
        logger.info(f"user input: {user_input}")
        logger.info(f"Summarized input: {summarized_input}")

        if intent == "sql_injection":
            response = ""

        else:
            print("checking for user requested columns...")
            if summarized_input.user_requested_columns:

                quantitaive_data = summarized_input.quantitative_data
                qualitative_data = summarized_input.qualitative_data

                quantitaive_query = generate_quantitaive_serach_query(
                    quantitaive_data, driver.db_config.dataset_table_name, primary_key
                )
                # print(f"quantitaive_query: {quantitaive_query}")
                quantitative_ids_uncleaned = driver.execute_query(quantitaive_query)

                quantitative_ids = [item[0] for item in quantitative_ids_uncleaned]

                print(f"quantitative_ids: {quantitative_ids}")

                qualitative_ids = qualitative_search(
                    chroma_collections, qualitative_data, primary_key
                )

                print(f"qualitative_ids: {qualitative_ids}")

                if not quantitative_ids:
                    intersection_ids = qualitative_ids
                elif not qualitative_ids:
                    intersection_ids = quantitative_ids
                else:
                    intersection_ids = list(
                        set(quantitative_ids) & set(qualitative_ids)
                    )

                print(intersection_ids)

                final_query = f"select * from {driver.db_config.dataset_table_name} where {primary_key} in ({','.join(f'\"{id}\"' for id in intersection_ids)})"

                # columns_database = driver.database_columns()

                response = str(driver.execute_query(final_query))

                # response = f"{columns_database}\n\n{data_retreived}"

                # final_query = f"select {', '.join(summarized_input.user_requested_columns)} from {driver.db_config.dataset_table_name} where {primary_key} in ({','.join(f'\"{id}\"' for id in intersection_ids)})"
                #  data_retreived = str(driver.execute_query(final_query))
                #  response = f"{summarized_input.user_requested_columns}\n\n{data_retreived}"

                logger.info(f"response: {response}")

            else:
                response = ""

        # chat_history.append((user_input, response))
        return response

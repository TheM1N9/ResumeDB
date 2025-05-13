import json
import logging
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate

# from langchain_openai import ChatOpenAI, OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI

import chromadb

# Create a logger object
logger = logging.getLogger(__name__)

# Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.INFO)


@dataclass
class SummarizedInput:
    """Class to represent the summarized input."""

    summary: str
    quantitative_data: Dict[str, str]
    qualitative_data: Dict[str, str]
    user_requested_columns: List[str]
    user_intent: str


# class Query:

#     def __init__(self) -> None:
#         self.column_descriptions, self.numerical_columns, self.categorical_columns = retrieve_descriptions_and_types_from_db()


# Default system prompt for the LLM.
DEFAULT_SYSTEM_PROMPT = "You are a professional medical assistant, adept at handling inquiries related to medical products."


# Generates a prompt for the LLM based on the instruction and system prompt.
def get_prompt(instruction: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    """Generates the prompt for the LLM.

    Args:
        instruction (str): The instruction for the LLM.
        system_prompt (str, optional): The system prompt for the LLM. Defaults to DEFAULT_SYSTEM_PROMPT.

    Returns:
        str: The prompt for the LLM.

    """
    SYSTEM_PROMPT = f"<<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
    return f"[INST]{SYSTEM_PROMPT}{instruction}[/INST]"


# Function to identify qualitative and quantitative data and user intent
def summarize(
    user_input: str,
    chat_history: List[Tuple[str, str]],
    column_descriptions_dictionary: Dict[str, str],
    numerical_columns: List[str],
    categorical_columns: List[str],
    llm: Union[ChatGoogleGenerativeAI, GoogleGenerativeAI],
) -> SummarizedInput:
    """Summarizes the user input and returns the summary, quantitative data, and qualitative data, along with the user requested columns in a JSON format.

    Args:
        user_input (str): The user input.
        chat_history (list[(str, str)]): The chat history.
        column_descriptions (dict[str, str]): The column descriptions.
        numerical_columns (list[str]): The numerical columns.
        categorical_columns (list[str]): The categorical columns.
        llm (Union[ChatOpenAI, OpenAI]): The LLM object.(Contains the details of the language we are using.)

    Returns:
        dict: {
            "summary": str,
            "quantitative_data": {
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
            },
            "qualitative_data": {
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
                "column name : str" : "Data mentioned about that column by the user : str",
            },
            "user_requested_columns": list,
            "user_intent":str,
        }
    """

    column_descriptions = list(column_descriptions_dictionary.items())

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                ResumeDB Assistant - An AI-powered Resume Analysis System, created by M1N9. Your task is to breakdown the Job description and help the user with his quiries.
                You will receive a user input and the chat history. Your task is to:
                
                1. **Single-Word Queries**: If the user input is a single word or very short (e.g., one or two words), provide a direct response if possible. If the query is unclear, prompt the user to elaborate.
                - Example response: "It seems you're asking about something specific. Could you provide more details?"

                2. **Structured Analysis**: For all other inputs, analyze the user input and identify key details based on our available data and chat history.
                
                3. Summarize the input, classifying the data into qualitative and quantitative categories.
                
                4. Identify relevant columns from which we can provide an answer. Pay close attention to the user's intent and specific mentions of data columns:
                - Are they seeking information about products, medications, treatments, or other relevant categories?
                - If the user is seeking information about a product, also provide the URL of the product if available.
                - Look for explicit mentions of column names, synonyms, or phrases that indicate the type of information requested. If the user specifies certain attributes or metrics, consider these as user-requested columns.

                5. Classify the user's intent. Possible intents include: phatic_communication, sql_injection, profanity, and other.

                6. Split or user input and with help of chat history create quantitative_data and qualitative_data dictionaries.

                7. Output the result in a JSON format.

                8. Do not output any other information except the JSON. Do not add [OUT], [/OUT] to the output.(!important)

                9. Use previous data from the chat history for creating the output.
                
                The output JSON should have the following structure:
                ```json
                    "summary": "summary of the user input",
                    "quantitative_data":
                                        ` 
                                        "column name": "Data mentioned about that column by the user in current query or from the available chat history. Example- < 4",
                                        "column name": "Data mentioned about that column by the user in current query or from the available chat history. Example- > 6.215",
                                        "column name": "Data mentioned about that column by the user in current query or from the available chat history. Example- >= 3.14 or <= 2.718",
                                        `,
                    "qualitative_data": 
                                        ` 
                                        "column name": "Data mentioned about that column by the user in current query or from the available chat history",
                                        "column name": "Data mentioned about that column by the user in current query or from the available chat history",
                                        "column name": "Data mentioned about that column by the user in current query or from the available chat history",
                                        `,
                    "user_requested_columns": "List of columns the user wants data from. If none, leave it as an empty list. Always add product and url to this column.",
                    "user_intent": "The user's intent. If none, leave it as an empty string.",
                ```
                
                The data we have and chat history:
                Data:{column_descriptions}\n\n 
                numerical columns in the data: {numerical_columns}\n\n 
                descriptive columns in the data: {categorical_columns}\n\n 
                Columns in the database: {columns}\n\n 
                chat history: {chat_history}

                Now, summarize the user input and provide the structured output in JSON format.
                """,
            ),
            ("human", "{user_input}"),
        ]
    )

    # print(f"prompt: {prompt}")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    summarized_input_str = str(
        chain.invoke(
            {
                "column_descriptions": column_descriptions,
                "numerical_columns": numerical_columns,
                "categorical_columns": categorical_columns,
                "columns": numerical_columns + categorical_columns,
                "chat_history": chat_history,
                "user_input": user_input,
            }
        )
    )

    print("------------------------------------------------------------------------")

    print(f"summarized_input_str: {summarized_input_str}")

    try:
        summarized_input_str = re.sub(r"```json|```", "", summarized_input_str)
        # Attempt to parse the summarized input as JSON
        summarized_input_dict = json.loads(summarized_input_str)
    except json.JSONDecodeError:
        # If parsing fails, return an empty SummarizedInput
        summarized_input_dict = {}

    logger.info("--------------------------")
    logger.info(f"user input: {user_input}")
    logger.info(f"Summarized input: {summarized_input_dict}")

    summarized_input = SummarizedInput(
        summary=summarized_input_dict.get("summary", ""),
        quantitative_data=summarized_input_dict.get("quantitative_data", {}),
        qualitative_data=summarized_input_dict.get("qualitative_data", {}),
        user_requested_columns=summarized_input_dict.get("user_requested_columns", []),
        user_intent=summarized_input_dict.get("user_intent", ""),
    )

    return summarized_input


def generate_quantitaive_serach_query(
    quantitaive_data: Dict[str, str], table_name: str, primary_key: str
) -> str:
    """Creates an SQL query from a dictionary of quantitative data.

    Args:
        quantitaive_data (dict): A dictionary of quantitative data in the form {'column_name': 'condition'}.

    Returns:
        str: The generated SQL query.
    """
    if not quantitaive_data:
        return ""  # Return an empty string if the dictionary is empty

    query_parts = []
    for column, condition in quantitaive_data.items():
        # Handle different comparison operators
        if "<" in condition:
            operator = "<"
        elif ">" in condition:
            operator = ">"
        elif "<=" in condition:
            operator = "<="
        elif ">=" in condition:
            operator = ">="
        elif "=" in condition:
            operator = "="
        else:
            operator = "LIKE"  # Default to LIKE for other conditions

        # Extract the value from the condition
        value = condition.replace(operator, "").strip()

        # Construct the query part
        query_part = f"{column} {operator} {value}"
        query_parts.append(query_part)

    # Combine the query parts with AND
    query_constraints = " AND ".join(query_parts)

    query = f"select {primary_key} from {table_name} where {query_constraints}"
    return query


def qualitative_search(
    collection: chromadb.Collection, data: Dict[str, str], primary_key: str
) -> Optional[List[Any]]:
    """
    Performs a similarity search on a ChromaDB collection for qualitative data and returns matching primary keys.
    
    For each column-condition pair in the provided data, queries the collection for entries with a similarity score of at least 0.3. Collects and returns a list of unique primary key values from the matching results, or None if no matches are found.
    
    Args:
        collection: The ChromaDB collection to search.
        data: Dictionary mapping column names to qualitative search conditions.
        primary_key: The name of the primary key column to extract from results.
    
    Returns:
        A list of unique primary key values for results above the similarity threshold, or None if no matches are found.
    """
    results = []
    count_in_chroma = collection.count()
    print(f"Total elements in the collection {count_in_chroma}")
    threshold = 0.3

    # Validate we have data to search
    if count_in_chroma == 0:
        return None

    for column, condition in data.items():
        print(f"Searching in the {column}")
        # Ensure valid query with error handling
        try:
            query_result = collection.query(
                query_texts=condition,
                n_results=count_in_chroma,
                where={"column_name": column},
            )

            distances = query_result.get("distances", [])
            num_matches = len(distances[0]) if distances and len(distances) > 0 else 0
            print(f"Query results for column '{column}': {num_matches} matches")

            # Get metadatas and distances
            metadatas = query_result.get("metadatas", [])
            distances = query_result.get("distances", [])

            # Process results if we have both metadata and distances
            if metadatas and distances and len(metadatas) > 0:
                # ChromaDB returns a list with one element per query
                for i in range(len(metadatas[0])):
                    # Convert distance to similarity score (0-1)
                    similarity = 1 - (distances[0][i] / 2)

                    if similarity >= threshold:
                        results.append(metadatas[0][i])

        except Exception as e:
            print(f"Error searching column '{column}': {e}")
            continue

    # Extract unique primary keys from filtered results
    ids = []
    seen = set()  # To track unique IDs

    for result in results:
        id_value = result.get(primary_key)
        if id_value and id_value not in seen:
            ids.append(id_value)
            seen.add(id_value)

    return ids if ids else None


# def qualitaive_search(collection: chromadb.Collection, data: Dict[str, str], primary_key: str) -> List[str]:
#     """Performs a similarity search on the database and returns all similar results.

#     Args:
#         collection (chromadb.Collection): The ChromaDB collection to search.
#         data (Dict[str, str]): A dictionary of qualitative data to search for.
#         primary_key (str): The primary key column name in the database.

#     Returns:
#         List[str]: A dictionary containing the search results.
#     """
#     all_ids = []

#     for column, condition in data.items():
#         query_result = collection.query(query_texts=condition, n_results=10, where={"column_name": column})

#         if query_result:
#             ids_for_column = set()  # Use a set to store unique IDs for this column
#             for result in query_result["metadatas"]:
#                 for item in result:
#                     id_value = item.get(primary_key)
#                     if id_value is not None:
#                         ids_for_column.add(str(id_value))  # Convert to string for comparison
#             all_ids.append(ids_for_column)

#     # Find the intersection of IDs across all columns
#     common_ids = set.intersection(*all_ids) if all_ids else set()

#     return list(common_ids)


# Function to perform a similarity search
def similarity_search(collection: chromadb.Collection, user_input: str) -> str:
    """Performs a similarity search on the database and returns the first similar result.

    Args:
        user_input (str): the user input.

    Returns:
        str: the first similar result.

    """
    result = collection.query(query_texts=user_input, n_results=1)
    if result:
        result_str = str(result)
        result_str = result_str.replace("{", "")
        result_str = result_str.replace("}", '"')
        logger.info(f"Result: {result_str}")
        return result_str
    else:
        logger.info("No similar result found.")
        return ""


# Function to generate a response based on the user input
def generate_query(
    user_input: str,
    summarized_input: SummarizedInput,
    chat_history: List[Tuple[str, str]],
    column_descriptions: Dict[str, str],
    numerical_columns: List[str],
    categorical_columns: List[str],
    llm: Union[ChatGoogleGenerativeAI, GoogleGenerativeAI],
    dataset_table_name: str,
) -> str:
    """Generates an SQL query based on the user input and chat history.

    Args:
        user_input (str): the user input.
        summarized_input (dict): the summarized input.
        chat_history (list[(str, str)]): the chat history.
        column_descriptions (dict): the column descriptions.
        numerical_columns (list[str]): the numerical columns.
        categorical_columns (list[str]): the categorical columns.
        llm (Union[ChatOpenAI, OpenAI]): the LLM object.
        dataset_table_name (str): the dataset table name.

    Returns:
        str: execute_query function executes the SQL query
    """
    quantitative_data = list(summarized_input.quantitative_data.items())
    qualitative_data = list(summarized_input.qualitative_data.items())
    user_requested_columns = summarized_input.user_requested_columns

    instruction = f"""
    Generate an SQLite query based on the user input and other data. For numerical columns, use exact matches. 
    For descriptive columns, use 'LIKE' for partial matches but handle possible spelling mistakes and close matches. 
    insert ORDER BY CustomerRating DESC LIMIT 3 if needed. 
    
    Generate the query according to the user input, chat history, and database schema. 
    Ensure that the query is robust, handles various user input scenarios, and incorporates appropriate conditions.
    Answer just the query without any explanation and code. 

    The data we have:
    numerical columns in the data: {numerical_columns}\n\n 
    descriptive columns in the data: {categorical_columns}\n\n 
    The columns in the database were {', '.join(column_descriptions.keys())}\n\n
    Table name: {dataset_table_name}\n\n
    User input: {user_input}\n\n
    quantitative data in the user input: {quantitative_data}\n\n
    qualitative data in the user input: {qualitative_data}\n\n
    user requested columns: {user_requested_columns}\n\n
    Chat history: {chat_history}\n\n

    Generate the SQLite query below:
    """

    system_prompt = "You are an expert in SQL queries. Create robust queries based on the user requirements and database schema."
    prompt = get_prompt(instruction, system_prompt)
    prompt_template = PromptTemplate(
        template=prompt, input_variables=["chat_history", "user_input"]
    )
    memory = ConversationBufferMemory(memory_key="chat_history")

    llm_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True, memory=memory)

    query = llm_chain.run(
        {"chat_history": chat_history, "user_input": user_input}
    ).strip()
    query = re.sub(r"```sql|```", "", query)
    logger.info(f"Query: {query}")

    return query

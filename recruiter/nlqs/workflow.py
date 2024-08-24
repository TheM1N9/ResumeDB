from typing import Dict, List, Tuple
import re
from recruiter.nlqs.database.sqlite import retrieve_descriptions_and_types_from_db, execute_query, validate_query
from recruiter.nlqs.query import generate_response, similarity_search, summarize, generate_query
from recruiter.scripts.description_generator import generate_column_description

# chroma_collections = get_chroma_collections()

column_descriptions, numerical_columns, categorical_columns = retrieve_descriptions_and_types_from_db()

if column_descriptions == {}:
    generate_column_description()
    column_descriptions, numerical_columns, categorical_columns = retrieve_descriptions_and_types_from_db()

def main_workflow(user_input:str, chat_history:List[Tuple[str, str]], column_descriptions_dict:Dict[str,str]=column_descriptions, numerical_columns_list:List[str]=numerical_columns, categorical_columns_list:List[str]=categorical_columns) -> Tuple[str,List[Tuple[str, str]]]:
    """This function is where the whole interaction happens. 
    It takes the user input and chat history as input and returns the response if the user's intent is either phatic_communication, profanity or sql_injection. 
    Else it returns the query result or search similarity result and the updated chat history.

    Args:
        user_input (str): The user's input.
        chat_history (list[(str, str)]): The chat history.
        column_descriptions_dict (dict[str, str]): The column descriptions.
        numerical_columns_list (list[str]): The numerical columns.
        categorical_columns_list (list[str]): The categorical columns.

    Returns:
        Tuple[str,List[Tuple[str, str]]]: The response and the updated chat history.
    """

    if not user_input.strip():
        response = ""

    user_input = re.sub(r"{|}", "", user_input)
    summarized_input = summarize(user_input, chat_history, column_descriptions_dict, numerical_columns_list, categorical_columns_list)

    count = 0
    print(f"summarized_input: {summarized_input}")
    while not summarized_input.summary and count < 5:
        summarized_input = summarize(user_input, chat_history, column_descriptions_dict, numerical_columns_list, categorical_columns_list)
        count += 1
        if count == 5:
            response = "Summarization failed. Please try again."
            break

    intent = summarized_input.user_intent
    
    if intent == "sql_injection" or intent == "profanity":
        response = generate_response(user_input=user_input, chat_history=chat_history, query_result="")
    
    else:
        if summarized_input.user_requested_columns:
            genenerted_query = generate_query(user_input, summarized_input, chat_history, column_descriptions_dict, numerical_columns_list, categorical_columns_list)
            if validate_query(genenerted_query):            
                query_result = execute_query(genenerted_query)
                if query_result == str([]):
                    query_result = similarity_search(user_input)
            else:
                query_result = "error while generating query. Please try again."
        else:
            query_result = ""

        print(query_result)

        response = generate_response(user_input=user_input, chat_history=chat_history, query_result=query_result)

    chat_history.append((user_input,response))
    return "", chat_history
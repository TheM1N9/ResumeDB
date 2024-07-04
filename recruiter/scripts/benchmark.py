"""
    Generates benchmarks against the testcases.csv file and stores the results in benchmark_results.csv.
"""

import csv
import os
import re
from nlqs.database.sqlite import execute_query, retrieve_descriptions_and_types_from_db, validate_query
from nlqs.query import get_chroma_instance, summarize, generate_query, similarity_search

# CSV file paths
TEST_CASES_FILE = "test_cases.csv"
BENCHMARK_RESULTS_FILE = "benchmark_results.csv"

data_vectors = get_chroma_instance()

# Main chat function
def chat_benchmark(user_input, chat_history):
    if not user_input:
        return chat_history, "", []
    
    column_descriptions, numerical_columns, categorical_columns = retrieve_descriptions_and_types_from_db()

    user_input = re.sub(r"{|}", "", user_input)
    summarized_input = summarize(user_input, chat_history, column_descriptions, numerical_columns, categorical_columns)

    if not summarized_input:
        summarized_input = summarize(user_input, chat_history, column_descriptions, numerical_columns, categorical_columns)
        
    if not summarized_input:
        response = "unable to summarize the data."
        log_data = [
        user_input, 
        "none",
        "none", 
        "none",
        "none",
        "none",
        "",  # Placeholder for query
        "",  # Placeholder for query result
        ""   # Placeholder for response
    ]
        return chat_history, response, log_data

    intent = summarized_input.user_intent    
    log_data = [ 
        user_input, 
        summarized_input.summary,
        summarized_input.quantitative_data,
        summarized_input.qualitative_data,
        summarized_input.user_requested_columns,
        intent, 
        "",  # Placeholder for query
        "",  # Placeholder for query result
        ""   # Placeholder for response
    ]

    if intent == "phatic_communication" or intent == "sql_injection" or intent == "profanity":
        response = "Sorry, I cannot process this request."
        chat_history.append((user_input, response))
        log_data[8] = response
        return chat_history, response, log_data
    
    if summarized_input.user_requested_columns:
        genenerted_query = generate_query(user_input, summarized_input, chat_history, column_descriptions, numerical_columns, categorical_columns)
        if validate_query(genenerted_query):
            query_result = execute_query(genenerted_query)
            log_data[6] = genenerted_query

            if query_result == str([]):
                similarity_result = similarity_search(data_vectors, user_input)
                response = "Similar data fetched."
                log_data[7] = similarity_result
            else:
                response = "Data fetched from database."
                log_data[7] = query_result
        else:
            log_data[6] = "none"
            response = "Sorry, I cannot process this request."
            log_data[7] = "none"          
    else:
        response = "Sorry, I cannot process this request."
        chat_history.append((user_input, response))
    
    log_data[8] = response
    chat_history.append((user_input, response))
    return chat_history, response, log_data

# Function to log data to CSV
def log_to_csv(log_data, file_path=BENCHMARK_RESULTS_FILE):
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["user_input", "summary", "quantitative_data", "qualitative_data", "user_requested_columns", "intent", "query", "query_result", "response"])
        writer.writerow(log_data)

# Run benchmark
def run_benchmark(test_cases_file=TEST_CASES_FILE, results_file=BENCHMARK_RESULTS_FILE):
    with open(test_cases_file, mode='r') as file:
        reader = csv.DictReader(file)
        chat_history = []
        for row in reader:
            interaction_type = row["interaction_type"]
            user_input = row["user_input"]
            chat_history, response, log_data = chat_benchmark(user_input, chat_history)
            log_to_csv(log_data, results_file)
            print(f"Interaction Type: {interaction_type}, User Input: {user_input}, Response: {response}")

if __name__ == "__main__":
    run_benchmark()

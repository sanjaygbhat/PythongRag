import os
from neo4j import GraphDatabase
from openai import OpenAI
import time
from dotenv import load_dotenv
import json
from app import initialize_neo4j_driver, close_neo4j_connection, llama_call
import logging

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Neo4j connection setup
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Global variables for tracking
total_input_tokens = 0
total_output_tokens = 0
total_cost = 0
total_time = 0

def calculate_cost(model, input_tokens, output_tokens):
    # Update these rates as needed
    rates = {
        "gpt-4o": {
            "input": 5 / 1000000,  # $5 per 1M tokens
            "output": 15 / 1000000  # $15 per 1M tokens
        }
    }
    model_rates = rates.get(model, {"input": 0, "output": 0})
    input_cost = input_tokens * model_rates["input"]
    output_cost = output_tokens * model_rates["output"]
    return input_cost + output_cost

def log_usage(model, input_tokens, output_tokens, cost, time_taken):
    global total_input_tokens, total_output_tokens, total_cost, total_time
    total_input_tokens += input_tokens
    total_output_tokens += output_tokens
    total_cost += cost
    total_time += time_taken
    print(f"API Call: {model}")
    print(f"  Input Tokens: {input_tokens}, Output Tokens: {output_tokens}")
    print(f"  Cost: ${cost:.6f}, Time: {time_taken:.2f}s")
    print(f"Totals - Input Tokens: {total_input_tokens}, Output Tokens: {total_output_tokens}")
    print(f"  Cost: ${total_cost:.6f}, Time: {total_time:.2f}s")

def query_gpt4o(prompt, max_tokens=4000, max_attempts=3):
    print("\nPrompt being sent to GPT-4:")
    print(json.dumps(prompt, indent=2))
    print("-" * 50)

    full_response = ""
    attempts = 0
    start_time = time.time()  # Initialize start_time

    while attempts < max_attempts:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=prompt,
                max_tokens=max_tokens,
                n=1,
                stop=None,
                temperature=0
            )

            chunk = response.choices[0].message.content.strip()
            full_response += chunk

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = calculate_cost("gpt-4o", input_tokens, output_tokens)
            time_taken = time.time() - start_time
            log_usage("gpt-4o", input_tokens, output_tokens, cost, time_taken)

            if len(chunk) < max_tokens:
                break

            attempts += 1

        except Exception as e:
            print(f"Error occurred: {e}")
            attempts += 1
            time.sleep(5)

    return full_response.strip()

def retrieve_graph_data(driver):
    with driver.session() as session:
        result = session.run("""
            MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
            RETURN a.name AS entity1, b.name AS entity2, r.type AS relationship
        """)
        return [(record["entity1"], record["entity2"], record["relationship"]) for record in result]

def filter_relevant_data(graph_data, query):
    prompt = [
        {"role": "system", "content": "You are an AI assistant that helps filter relevant information from graph data based on a given query. Always respond with a valid JSON array."},
        {"role": "user", "content": f"""Given the following graph data and query, return only the relevant information that could be used to answer the query. Format your response as a JSON array of objects.

Graph Data:
{json.dumps(graph_data, indent=2)}

Query: {query}

Return the filtered data as a JSON array of objects. Ensure your entire response is a valid JSON array."""}
    ]
    response = llama_call(prompt)
    
    # Try to parse the response as JSON
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # If parsing fails, try to extract JSON from the response
        try:
            json_start = response.index('[')
            json_end = response.rindex(']') + 1
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            # If extraction fails, log the error and return an empty list
            logging.error(f"Failed to parse LLM response as JSON. Response: {response}")
            return []

import functools
from collections import OrderedDict

# LRU Cache for memoization
class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Initialize the cache
llama_cache = LRUCache(100)  # Adjust the capacity as needed

@functools.lru_cache(maxsize=100)
def extract_relevant_text(query, combined_text):
    # Check if the result is in the cache
    cache_key = (query, combined_text)
    cached_result = llama_cache.get(cache_key)
    if cached_result is not None:
        print("Cache hit for LLaMA 3 query")
        return cached_result

    prompt = [
        {"role": "system", "content": "Extract only the text that is directly relevant to the given query from the provided information. Do not add or infer any information."},
        {"role": "user", "content": f"""Query: {query}

Text: {combined_text}

Extract and return only the relevant portions of the text."""}
    ]
    print("\nLLaMA 3 Prompt:")
    print(json.dumps(prompt, indent=2))
    response = llama_call(prompt)
    print("\nLLaMA 3 Response:")
    print(response)

    # Store the result in the cache
    llama_cache.put(cache_key, response.strip())

    return response.strip()

def generate_answers_from_graph(driver, query):
    graph_data = retrieve_graph_data(driver)
    relevant_data = filter_relevant_data(graph_data, query)
    relevant_texts = []
    
    with driver.session() as session:
        for entity1, entity2, relationship in relevant_data:
            cypher_query = """
            MATCH (a:Entity {name: $entity1})-[r:RELATES_TO {type: $relationship}]->(b:Entity {name: $entity2})
            RETURN a.content AS entity1_content, b.content AS entity2_content
            """
            result = session.run(cypher_query, entity1=entity1, entity2=entity2, relationship=relationship)
            record = result.single()
            if record:
                entity1_content = json.loads(record["entity1_content"]) if record["entity1_content"] else {}
                entity2_content = json.loads(record["entity2_content"]) if record["entity2_content"] else {}
                
                entity1_text = entity1_content.get(relationship, [""])[0]
                entity2_text = entity2_content.get(relationship, [""])[0]
                
                combined_text = f"{entity1}: {entity1_text}\n{entity2}: {entity2_text}"
                
                relevant_text = extract_relevant_text(query, combined_text)
                if relevant_text.strip():
                    relevant_texts.append(f"Information about {entity1} and {entity2}:\n{relevant_text}")
    
    combined_text = "\n\n".join(relevant_texts)
    
    gpt4_prompt = [
        {"role": "system", "content": "You are an expert in government tenders. Provide a comprehensive and detailed answer to the query based on the provided information."},
        {"role": "user", "content": f"""Query: {query}

Relevant Information:
{combined_text}

Please provide a comprehensive and detailed answer to the query based on the given information."""}
    ]
    
    final_answer = query_gpt4o(gpt4_prompt)
    
    return final_answer

def answer_query(driver, user_query):
    return generate_answers_from_graph(driver, user_query)

if __name__ == "__main__":
    try:
        driver = initialize_neo4j_driver()
        while True:
            user_query = input("Enter your question (or 'quit' to exit): ")
            if user_query.lower() == 'quit':
                break

            answer = answer_query(driver, user_query)
            print(f"Answer: {answer}\n")

    finally:
        close_neo4j_connection(driver)
        print("\nFinal Usage Statistics:")
        print(f"Total Input Tokens: {total_input_tokens}")
        print(f"Total Output Tokens: {total_output_tokens}")
        print(f"Total Cost: ${total_cost:.6f}")
        print(f"Total Time: {total_time:.2f}s")


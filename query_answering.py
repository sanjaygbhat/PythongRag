import os
from neo4j import GraphDatabase
from openai import OpenAI
import time
from dotenv import load_dotenv
import json
from app import initialize_neo4j_driver, close_neo4j_connection

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

def query_gpt4o(prompt, max_tokens=30000, max_attempts=10):
    print("Prompt being sent to GPT-4:")
    print(prompt)
    print("-" * 50)

    full_response = ""
    attempts = 0
    remaining_tokens = max_tokens
    start_time = time.time()

    while attempts < max_attempts and remaining_tokens > 0:
        tokens_this_iteration = min(remaining_tokens, 2000)

        if isinstance(prompt, list):
            messages = prompt + [{"role": "user", "content": "Continue: " + full_response}] if full_response else prompt
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided graph data."},
                {"role": "user", "content": prompt + (" Continue: " + full_response if full_response else "")}
            ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=tokens_this_iteration,
                n=1,
                stop=None,
                temperature=0.7
            )

            chunk = response.choices[0].message.content.strip()
            full_response += chunk

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = calculate_cost("gpt-4o", input_tokens, output_tokens)
            time_taken = time.time() - start_time
            log_usage("gpt-4o", input_tokens, output_tokens, cost, time_taken)

            remaining_tokens -= (input_tokens + output_tokens)
            attempts += 1

            if chunk.endswith((".", "!", "?")) and len(chunk) > 100:
                break

            if len(chunk) < 50:
                break

        except Exception as e:
            print(f"Error occurred: {e}")
            attempts += 1
            time.sleep(5)

    return full_response.strip()

def retrieve_graph_data(driver, query):
    with driver.session() as session:
        result = session.run("""
            MATCH (cs:CommunitySummaries)
            RETURN cs.data AS summaries
        """)
        community_summaries = json.loads(result.single()["summaries"])
        return community_summaries

def format_graph_data(graph_data):
    formatted_data = []
    for item in graph_data:
        entity = item['entity']
        relationships = item['relationships']
        formatted_relationships = [f"{entity} {rel['relation']} {rel['target']}" for rel in relationships if rel['target']]
        formatted_data.append(f"Entity: {entity}")
        formatted_data.extend(formatted_relationships)
    return "\n".join(formatted_data)

def classify_query_intent(query):
    response = query_gpt4o([
        {"role": "system", "content": "You are an expert in classifying query intents for government tender documents. Categorize the following query into one of these types: Qualification, Submission, Deadline, Requirements, Evaluation, or General."},
        {"role": "user", "content": f"Classify the intent of this query: {query}"}
    ])
    return response.strip()

def expand_query(query, intent):
    response = query_gpt4o([
        {"role": "system", "content": "You are an expert in government tender terminology. Expand the given query with relevant synonyms and related terms specific to government tenders."},
        {"role": "user", "content": f"Expand this query with relevant terms. Query intent: {intent}\nQuery: {query}"}
    ])
    return response.strip()

def generate_answers_from_communities(driver, community_summaries, query, intent):
    intermediate_answers = []
    for index, summary in enumerate(community_summaries[:5]):
        print(f"Summary index {index} of {len(community_summaries)}:")
        
        if isinstance(summary, str):
            try:
                summary = json.loads(summary)
            except json.JSONDecodeError:
                summary = {"summary": summary}
        elif not isinstance(summary, dict):
            summary = {"summary": str(summary)}
        
        if 'summary' not in summary:
            summary['summary'] = str(summary)
        
        # Retrieve the original text for entities in this community
        with driver.session() as session:
            result = session.run("""
                MATCH (e:Entity)
                WHERE e.name IN $members
                RETURN e.name AS name, e.text AS text
            """, members=summary['members'])
            entity_texts = {record["name"]: (record["text"] if record["text"] is not None else "") for record in result}
        
        community_text = "\n".join(entity_texts.values())[:1000]
        
        messages = [
            {"role": "system", "content": f"Expert in government tenders. Answer based on provided info. Intent: {intent}."},
            {"role": "user", "content": f"Query: {query}\nCommunity Summary: {summary['summary']}\nOriginal Text: {community_text}"}
        ]
        response = query_gpt4o(messages)
        print("Intermediate answer:", response)
        intermediate_answers.append(response[:500])

    final_messages = [
        {"role": "system", "content": f"You are an expert in government tenders. Combine these answers into a final, comprehensive response. The query intent is: {intent}. Ensure all relevant information is included and the answer directly addresses the original query."},
        {"role": "user", "content": f"Query: {query}\nIntermediate answers: {intermediate_answers}"}
    ]
    final_answer = query_gpt4o(final_messages)
    return final_answer

def answer_query(driver, user_query):
    intent = classify_query_intent(user_query)
    expanded_query = expand_query(user_query, intent)
    community_summaries = retrieve_graph_data(driver, expanded_query)
    return generate_answers_from_communities(driver, community_summaries, expanded_query, intent)

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

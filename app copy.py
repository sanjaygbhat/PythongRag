import os
import time
import json
import requests
import logging
import random
from dotenv import load_dotenv
from neo4j import GraphDatabase
from constants import DOCUMENTS
import backoff

load_dotenv()

DEEPINFRA_API_TOKEN = os.getenv("DEEPINFRA_API_TOKEN")

# Ensure the token is set
if not DEEPINFRA_API_TOKEN:
    raise ValueError("DEEPINFRA_API_TOKEN is not set in the .env file")

API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

# Neo4j connection setup
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def initialize_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

driver = initialize_neo4j_driver()

# Global variables to track metrics
total_tokens_consumed = 0
total_cost = 0
total_time = 0
call_count = 0

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("output.log"),
                        logging.StreamHandler()
                    ])

def reset_metrics():
    global total_tokens_consumed, total_cost, total_time, call_count
    total_tokens_consumed = 0
    total_cost = 0
    total_time = 0
    call_count = 0
    save_metrics()

def load_metrics():
    global total_tokens_consumed, total_cost, total_time, call_count
    if os.path.exists('metrics.json'):
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
        total_tokens_consumed = metrics['total_tokens']
        total_cost = metrics['total_cost']
        total_time = metrics['total_time']
        call_count = metrics['call_count']
    else:
        reset_metrics()

def save_metrics():
    metrics = {
        'total_tokens': total_tokens_consumed,
        'total_cost': total_cost,
        'total_time': total_time,
        'call_count': call_count
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

@backoff.on_exception(backoff.expo, (requests.exceptions.RequestException, KeyError), max_tries=5)
def llama_call(messages, max_tokens=8192):
    global total_tokens_consumed, total_cost, total_time, call_count
    
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_TOKEN}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens
    }
    
    start_time = time.time()
    response = requests.post(API_URL, json=data, headers=headers)
    response.raise_for_status()
    result = response.json()
    
    if 'choices' in result and len(result['choices']) > 0:
        generated_text = result['choices'][0]['message']['content']
        tokens = result['usage']['total_tokens']
        call_cost = result['usage']['estimated_cost']
        
        total_tokens_consumed += tokens
        total_cost += call_cost
        call_time = time.time() - start_time
        total_time += call_time
        call_count += 1

        logging.info(f"\nLLaMA 3 Call {call_count}:")
        logging.info(f"Output: {generated_text}")
        logging.info(f"Tokens: {tokens}")
        logging.info(f"Cost: ${call_cost:.6f}")
        logging.info(f"Time: {call_time:.2f} seconds")
        logging.info(f"\nCumulative Metrics:")
        logging.info(f"Total Tokens: {total_tokens_consumed}")
        logging.info(f"Total Cost: ${total_cost:.6f}")
        logging.info(f"Total Time: {total_time:.2f} seconds")

        return generated_text
    else:
        raise KeyError("Unable to find generated text in API response")

def split_document_into_chunks(file_path, chunk_size=600, overlap_size=100):
    chunks = []
    with open(file_path, 'r') as file:
        current_chunk = []
        current_size = 0
        for line in file:
            current_chunk.append(line)
            current_size += len(line)
            if current_size >= chunk_size:
                chunks.append(''.join(current_chunk))
                current_chunk = current_chunk[-overlap_size:]
                current_size = sum(len(line) for line in current_chunk)
    if current_chunk:
        chunks.append(''.join(current_chunk))
    return chunks

def process_chunk(chunk, chunk_id):
    prompt = [
        {"role": "system", "content": "Extract entities and relationships from the given text. Provide the specific text that supports each relationship."},
        {"role": "user", "content": f"""Extract entities and relationships from the following text. Format your response as follows:

Entities:
1. Entity1
2. Entity2
...

Relationships:
1. Entity1 -> Relation -> Entity2
   Text: "Exact text from the chunk that supports this relationship"
2. Entity3 -> Relation -> Entity4
   Text: "Exact text from the chunk that supports this relationship"
...

Text:
{chunk}"""}
    ]
    response = llama_call(prompt)
    return {"chunk_id": chunk_id, "content": chunk, "extracted": response}

def create_or_merge_node(tx, entity_name, relation_id, chunk_content):
    query = """
    MERGE (e:Entity {name: $name})
    SET e.content = $new_content
    RETURN e
    """
    
    # First, get the current content
    result = tx.run("MATCH (e:Entity {name: $name}) RETURN e.content AS content", name=entity_name)
    record = result.single()
    current_content = record['content'] if record else None
    
    # Parse the current content or create a new dictionary
    if current_content:
        try:
            content_dict = json.loads(current_content)
        except json.JSONDecodeError:
            content_dict = {}
    else:
        content_dict = {}
    
    # Update the content
    if relation_id in content_dict:
        if isinstance(content_dict[relation_id], list):
            content_dict[relation_id].append(chunk_content)
        else:
            content_dict[relation_id] = [content_dict[relation_id], chunk_content]
    else:
        content_dict[relation_id] = [chunk_content]
    
    # Convert back to JSON string
    new_content = json.dumps(content_dict)
    
    # Run the query to update the node
    result = tx.run(query, name=entity_name, new_content=new_content)
    return result.single()['e']

def store_chunk_in_graph(session, chunk):
    entities = []
    relationships = []
    current_section = None
    current_relationship = None
    
    for line in chunk['extracted'].split('\n'):
        if line.startswith("Entities:"):
            current_section = "entities"
        elif line.startswith("Relationships:"):
            current_section = "relationships"
        elif current_section == "entities" and line.strip():
            entities.append(line.strip().split(". ", 1)[-1])
        elif current_section == "relationships":
            if " -> " in line:
                current_relationship = line.strip()
            elif line.strip().startswith("Text:"):
                relationship_text = line.strip()[5:].strip()
                relationships.append((current_relationship, relationship_text))
    
    # Create nodes with chunk information
    for entity in entities:
        session.execute_write(create_or_merge_node, entity, f"entity_{chunk['chunk_id']}", chunk['content'])
    
    # Update nodes with relationship information and create relationships
    for rel, text in relationships:
        parts = rel.split(" -> ")
        if len(parts) == 3:
            source, relation, target = parts
            source = source.split(". ", 1)[-1]
            target = target.split(". ", 1)[-1]
            relation_id = f"{relation}"
            
            # Update source node
            session.execute_write(create_or_merge_node, source, relation_id, text)
            
            # Update target node
            session.execute_write(create_or_merge_node, target, relation_id, text)
            
            # Create relationship
            session.execute_write(create_relationship, source, target, relation, text, relation_id)

def create_relationship(tx, source, target, relation, text, relation_id):
    query = """
    MATCH (a:Entity {name: $source})
    MATCH (b:Entity {name: $target})
    MERGE (a)-[r:RELATES_TO {type: $relation}]->(b)
    SET r.text = $text,
        r.relation_id = $relation_id
    RETURN r
    """
    result = tx.run(query, source=source, target=target, relation=relation, text=text, relation_id=relation_id)
    single_result = result.single()
    if single_result is None:
        return None
    return single_result['r']

def process_and_store_chunk(driver, chunk, chunk_id):
    processed_chunk = process_chunk(chunk, chunk_id)
    with driver.session() as session:
        store_chunk_in_graph(session, processed_chunk)

def graph_rag_pipeline(input_file, chunk_size=600, overlap_size=100):
    reset_metrics()  # Reset metrics at the start of each run
    
    chunks = split_document_into_chunks(input_file, chunk_size, overlap_size)
    
    driver = initialize_neo4j_driver()
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")  # Clear existing data
        logging.info("Cleared existing data from Neo4j database")
        
        for i, chunk in enumerate(chunks):
            process_and_store_chunk(driver, chunk, i)
            logging.info(f"Processed and stored chunk {i+1}/{len(chunks)}")
        
        save_metrics()  # Save final metrics at the end of the pipeline
    finally:
        close_neo4j_connection(driver)

    return driver

def close_neo4j_connection(driver):
    if driver:
        driver.close()
    logging.info("Neo4j connection closed.")

if __name__ == "__main__":
    input_file = "example_graphrag/example_text/input.txt"
    
    print(f"Attempting to read file from: {input_file}")
    
    overall_start_time = time.time()
    
    driver = graph_rag_pipeline(input_file)
    
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    
    logging.info(f"\nTotal LLaMA 3 Calls: {call_count}")
    logging.info(f"Total Tokens: {total_tokens_consumed}")
    logging.info(f"Total Cost: ${total_cost:.6f}")
    logging.info(f"Total API Call Time: {total_time:.2f} seconds")
    logging.info(f"Overall Processing Time: {overall_time:.2f} seconds")

    close_neo4j_connection(driver)

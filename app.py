import os
import time
import json
import requests
import logging
import random
from dotenv import load_dotenv
from neo4j import GraphDatabase
from constants import DOCUMENTS

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


def llama_call(messages, max_tokens=8192):
    global total_tokens_consumed, total_cost, total_time, call_count
    
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_TOKEN}",
        "Content-Type": "application/json",
    }
    
    data = {
        "model": "meta-llama/Meta-Llama-3-70B-Instruct",
        "messages": messages
    }
    
    print(f"API URL: {API_URL}")
    print(f"Headers: {json.dumps(headers, indent=2)}")
    print(f"Data: {json.dumps(data, indent=2)}")
    
    start_time = time.time()
    try:
        response = requests.post(API_URL, json=data, headers=headers)
        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {json.dumps(dict(response.headers), indent=2)}")
        print(f"Response content: {response.text}")
        response.raise_for_status()
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        if hasattr(e, 'response'):
            print(f"Error response content: {e.response.text}")
        raise

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

def split_documents_into_chunks(documents, chunk_size=600, overlap_size=100):
    chunks = []
    for document in documents:
        for i in range(0, len(document), chunk_size - overlap_size):
            chunk = document[i:i + chunk_size]
            chunks.append(chunk)
    return chunks

def extract_elements_from_chunks(chunks):
    elements = []
    for index, chunk in enumerate(chunks):
        response = llama_call([
            {"role": "system", "content": "You are an expert in Named Entity Recognition and Relationship Extraction, specifically for government tender documents. Extract only explicitly mentioned entities and relationships from the following text."},
            {"role": "user", "content": f"Extract entities and relationships from the following text. Only include explicitly mentioned information. Format your response as follows:\n\nEntities:\n1. Entity1 (Type: Person/Organization/Document/Requirement/etc.)\n2. Entity2 (Type: Person/Organization/Document/Requirement/etc.)\n...\n\nRelationships:\n1. Entity1 -> Relation -> Entity2\n2. Entity3 -> Relation -> Entity4\n...\n\nText:\n{chunk}"}
        ])
        elements.append(response)
    return elements

def summarize_elements(elements):
    summaries = []
    for element in elements:
        prompt = f"""
Summarize the following text into a structured format of entities and relationships. Follow these guidelines strictly:

1. Identify only explicitly mentioned entities (nouns or noun phrases) in the text and list them under "Entities".

2. Identify only explicitly stated relationships between these entities and list them under "Relationships".

3. Each relationship should be in the format: Entity A -> relationship -> Entity B.

4. Ensure that each relationship points to a single entity only

5. Do not infer or imply any entities or relationships not directly stated in the text.

6. Ensure that all entities mentioned in relationships are also listed in the Entities section.

Text to summarize:
{element}

Provide your summary in this format:
**Entities**
1. Entity1
2. Entity2
...

**Relationships**
1. EntityA -> relationship -> EntityB
2. EntityC -> relationship -> EntityD
...
"""
        response = llama_call([
            {"role": "system", "content": "You are a helpful assistant that summarizes text into entities and relationships. Only include explicitly mentioned information."},
            {"role": "user", "content": prompt}
        ])
        summaries.append(response)
    return summaries

def create_or_merge_node(tx, entity_name, text):
    query = """
    MERGE (e:Entity {name: $name})
    SET e.text = $text
    RETURN e
    """
    result = tx.run(query, name=entity_name, text=text)
    return result.single()['e']

def create_or_get_node(tx, entity_name):
    query = """
    MERGE (e:Entity {name: $name})
    RETURN e
    """
    result = tx.run(query, name=entity_name)
    return result.single()['e']

def create_relationship(tx, source, target, relation):
    query = """
    MERGE (a:Entity {name: $source})
    MERGE (b:Entity {name: $target})
    MERGE (a)-[r:RELATES_TO {label: $relation}]->(b)
    RETURN r
    """
    result = tx.run(query, source=source, target=target, relation=relation)
    record = result.single()
    if record and record.get('r'):
        return record['r']
    else:
        logging.warning(f"Failed to create relationship: {source} -> {relation} -> {target}")
        return None

def build_graph_from_summaries(summaries_json, driver):
    summaries = json.loads(summaries_json)
    
    # Initialize relationship_types dictionary
    relationship_types = {}
    
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")  # Clear existing data
        logging.info("Cleared existing data from Neo4j database")
        
        node_count = 0
        relationship_count = 0
        
        for index, summary in enumerate(summaries):
            logging.info(f"Processing summary {index + 1} of {len(summaries)}")
            lines = summary.split("\n")
            entities_section = False
            relationships_section = False
            entities = []  # Change this from set to list
            
            for line in lines:
                if "**Entities**" in line or "**Entities:**" in line:
                    entities_section = True
                    relationships_section = False
                    continue
                elif "**Relationships**" in line or "**Relationships:**" in line:
                    entities_section = False
                    relationships_section = True
                    continue
                
                if entities_section and line.strip():
                    entity = line.split(".", 1)[-1].strip().strip('*')
                    entities.append(entity)  # Use append instead of add
                    session.execute_write(create_or_merge_node, entity, summary)  # Pass the summary as text
                    node_count += 1
                
                elif relationships_section and "->" in line:
                    parts = line.split("->")
                    if len(parts) == 3:
                        source = parts[0].split(".", 1)[-1].strip()
                        relation = parts[1].strip()
                        target = parts[2].strip()

                        result = session.write_transaction(
                            create_relationship, 
                            source, 
                            target, 
                            relation
                        )
                        if result:
                            relationship_count += 1
                            relationship_types[relation] = relationship_types.get(relation, 0) + 1
    
    logging.info(f"Graph built with {node_count} nodes and {relationship_count} relationships")
    logging.info("Types of relationships created:")
    for rel_type, count in relationship_types.items():
        logging.info(f"  {rel_type}: {count}")

    return driver

def detect_communities(driver, algorithm='louvain', min_community_size=3):
    with driver.session() as session:
        if not check_relationships_exist(driver):
            logging.warning("No relationships found in the database. Cannot detect communities.")
            return []

        session.run("CALL gds.graph.drop('myGraph', false) YIELD graphName")

        # Modified graph projection without relationship properties
        result = session.run("""
            CALL gds.graph.project(
                'myGraph',
                'Entity',
                'RELATES_TO'
            )
            YIELD graphName, nodeCount, relationshipCount
        """)
        projection_info = result.single()
        logging.info(f"Graph projection created: {projection_info}")

        if projection_info["relationshipCount"] == 0:
            logging.warning("No relationships in the projected graph. Cannot detect communities.")
            return []

        if algorithm == 'louvain':
            result = session.run(f"""
                CALL gds.louvain.stream('myGraph', {{
                    includeIntermediateCommunities: true,
                    tolerance: 0.0001,
                    maxLevels: 10,
                    maxIterations: 20,
                    minCommunitySize: {min_community_size}
                }})
                YIELD nodeId, communityId, intermediateCommunityIds
                RETURN gds.util.asNode(nodeId).name AS name, communityId, intermediateCommunityIds
                ORDER BY communityId ASC
            """)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        communities = {}
        for record in result:
            community_id = record["communityId"]
            name = record["name"]
            if community_id not in communities:
                communities[community_id] = set()
            communities[community_id].add(name)

    logging.info(f"Detected {len(communities)} communities")
    for community_id, members in communities.items():
        logging.info(f"Community {community_id}: {len(members)} members")

    return communities

def merge_small_communities(communities, min_size=3):
    merged_communities = {}
    misc_community = []

    for community_id, members in communities.items():
        if len(members) >= min_size:
            merged_communities[community_id] = members
        else:
            misc_community.extend(members)

    if misc_community:
        merged_communities['misc'] = misc_community

    logging.info(f"After merging: {len(merged_communities)} communities")
    for community_id, members in merged_communities.items():
        logging.info(f"Community {community_id}: {len(members)} members")

    return merged_communities

def hierarchical_community_detection(driver, min_community_size=5, max_communities=20):
    communities = detect_communities(driver)
    
    # Sort communities by size in descending order
    sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
    
    # Keep track of merged communities
    merged_communities = {}
    unassigned_nodes = set()
    
    for community_id, members in sorted_communities:
        if len(members) >= min_community_size:
            merged_communities[community_id] = list(members)
        else:
            unassigned_nodes.update(members)
    
    # Assign unassigned nodes to the closest large community
    with driver.session() as session:
        for node in unassigned_nodes:
            result = session.run("""
                MATCH (n:Entity {name: $node})-[:RELATES_TO*1..2]-(m:Entity)
                WHERE m.name IN $community_members
                RETURN m.name AS neighbor, count(*) AS strength
                ORDER BY strength DESC
                LIMIT 1
            """, node=node, community_members=list(set([m for c in merged_communities.values() for m in c])))
            
            closest_community = result.single()
            if closest_community:
                for community_id, members in merged_communities.items():
                    if closest_community['neighbor'] in members:
                        members.append(node)
                        break
            else:
                # If no connection found, assign to the largest community
                next(iter(merged_communities.values())).append(node)
    
    # If we still have too many communities, merge the smallest ones
    while len(merged_communities) > max_communities:
        smallest_community = min(merged_communities.items(), key=lambda x: len(x[1]))
        del merged_communities[smallest_community[0]]
        next(iter(merged_communities.values())).extend(smallest_community[1])
    
    logging.info(f"Final number of communities after hierarchical detection: {len(merged_communities)}")
    return merged_communities


def create_or_get_node(tx, entity_name):
    query = (
        "MERGE (e:Entity {name: $name}) "
        "RETURN e"
    )
    result = tx.run(query, name=entity_name)
    return result.single()['e']

def check_relationship_count(driver):
    with driver.session() as session:
        result = session.run("MATCH ()-[r]->() RETURN COUNT(r) AS rel_count")
        rel_count = result.single()["rel_count"]
        logging.info(f"Direct database query: {rel_count} relationships found")
    return rel_count

def check_relationships_exist(driver):
    with driver.session() as session:
        result = session.run("MATCH ()-[r:RELATES_TO]->() RETURN COUNT(r) AS count")
        return result.single()["count"] > 0

def summarize_communities(communities, driver):
    community_summaries = []
    with driver.session() as session:
        for community_id, members in communities.items():
            logging.info(f"Summarizing community {community_id} with {len(members)} members")
            
            # Convert set to list
            members_list = list(members)
            
            # Sample relationships if the community is too large
            if len(members_list) > 100:
                sample_size = min(100, len(members_list) // 2)
                sampled_members = random.sample(members_list, sample_size)
            else:
                sampled_members = members_list
            
            result = session.run("""
                MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
                WHERE a.name IN $members AND b.name IN $members
                RETURN a.name AS source, b.name AS target, r.label AS relation
                LIMIT 1000
            """, members=sampled_members)
            
            relationships = [f"{r['source']} -> {r['relation']} -> {r['target']}" for r in result]
            
            description = f"Community {community_id}:\nEntities: {', '.join(list(members)[:100])}{'...' if len(members) > 100 else ''}\nRelationships: {'; '.join(relationships[:100])}{'...' if len(relationships) > 100 else ''}"
            
            logging.info(f"Description for community {community_id}:\n{description[:1000]}...")
            
            response = llama_call([
                {"role": "system", "content": "Summarize the following community of entities and relationships, focusing on the most important and relevant information for a government tender document."},
                {"role": "user", "content": description},
                {"role": "system", "content": "Provide a concise summary of the community, highlighting key entities and relationships that are most relevant to government tender processes."}
            ])
            
            logging.info(f"LLaMA response for community {community_id}:\n{response[:500]}...")
            
            community_summaries.append({
                "id": community_id,
                "members": list(members),
                "relationships": relationships[:100],
                "summary": response
            })
    
    return community_summaries

def process_and_store(summaries_json, driver):
    build_graph_from_summaries(summaries_json, driver)
    communities = hierarchical_community_detection(driver, min_community_size=5, max_communities=20)
    community_summaries = summarize_communities(communities, driver)
    
    with driver.session() as session:
        session.run("MERGE (cs:CommunitySummaries) SET cs.data = $summaries", 
                    summaries=json.dumps(community_summaries))
        logging.info("Stored community summaries in the database")

def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)
    logging.info(f"Saved data to {filename}")

def load_from_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    logging.info(f"Loaded data from {filename}")
    return data

def get_start_option():
    while True:
        choice = input("Do you want to start from scratch or continue from summaries? (scratch/continue): ").lower()
        if choice in ['scratch', 'continue']:
            return choice
        logging.info("Invalid choice. Please enter 'scratch' or 'continue'.")

def graph_rag_pipeline(documents, chunk_size=600, overlap_size=100, start_option='scratch'):
    reset_metrics()  # Reset metrics at the start of each run
    
    if start_option == 'scratch':
        chunks = split_documents_into_chunks(documents, chunk_size, overlap_size)
        elements = extract_elements_from_chunks(chunks)
        save_to_json(elements, 'elements.json')
        summaries = summarize_elements(elements)
        save_to_json(summaries, 'summaries.json')
    else:
        if os.path.exists('summaries.json'):
            summaries = load_from_json('summaries.json')
        else:
            logging.info("No saved summaries found. Starting from scratch.")
            chunks = split_documents_into_chunks(documents, chunk_size, overlap_size)
            elements = extract_elements_from_chunks(chunks)
            save_to_json(elements, 'elements.json')
            summaries = summarize_elements(elements)
            save_to_json(summaries, 'summaries.json')

    # Convert summaries to JSON string if it's a list
    if isinstance(summaries, list):
        summaries_json = json.dumps(summaries)
    else:
        summaries_json = summaries

    driver = initialize_neo4j_driver()
    try:
        process_and_store(summaries_json, driver)
        save_metrics()  # Save final metrics at the end of the pipeline
    finally:
        close_neo4j_connection(driver)

    return driver

def close_neo4j_connection(driver):
    if driver:
        driver.close()
    logging.info("Neo4j connection closed.")

if __name__ == "__main__":
    start_option = get_start_option()
    
    overall_start_time = time.time()
    
    driver = graph_rag_pipeline(DOCUMENTS, start_option=start_option)
    
    overall_end_time = time.time()
    overall_time = overall_end_time - overall_start_time
    
    logging.info(f"\nTotal LLaMA 3 Calls: {call_count}")
    logging.info(f"Total Tokens: {total_tokens_consumed}")
    logging.info(f"Total Cost: ${total_cost:.6f}")
    logging.info(f"Total API Call Time: {total_time:.2f} seconds")
    logging.info(f"Overall Processing Time: {overall_time:.2f} seconds")

    close_neo4j_connection(driver)





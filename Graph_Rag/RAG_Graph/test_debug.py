import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add Model directory to sys.path to ensure we can import Graph_Rag
sys.path.append(os.path.join(os.path.dirname(__file__), 'Model'))

try:
    from Graph_Rag import driver, generate_cypher_query_gemini, execute_cypher_query, generate_final_answer, system_prompt
except ImportError as e:
    print(f"Error importing Graph_Rag: {e}")
    # Fallback if running from root
    sys.path.append(os.path.join(os.getcwd(), 'RAG_Graph', 'Model'))
    from Graph_Rag import driver, generate_cypher_query_gemini, execute_cypher_query, generate_final_answer, system_prompt

def test_query(query_text):
    print(f"\n--- Testing Query: '{query_text}' ---")
    
    # 1. Generate Cypher
    cypher = generate_cypher_query_gemini(query_text, system_prompt)
    print(f"Generated Cypher:\n{cypher}")
    
    if not cypher:
        print("Failed to generate Cypher.")
        return

    # 2. Execute Cypher
    print("Executing Cypher...")
    context = execute_cypher_query(driver, cypher)
    print(f"Retrieved Context:\n{context}")

    # 3. Generate Answer
    print("Generating Final Answer...")
    # Mock client if needed, but we should rely on the one imported
    from Graph_Rag import client
    answer = generate_final_answer(client, query_text, context)
    print(f"Final Answer:\n{answer}")

def inspect_db():
    print("\n--- Inspecting Database ---")
    if driver:
        with driver.session() as session:
            result = session.run("MATCH (c:Company) RETURN c.name, c.ticker, c.sector LIMIT 5")
            print("First 5 Companies:")
            for record in result:
                print(record.data())
    else:
        print("Driver is not connected.")

if __name__ == "__main__":
    inspect_db()
    test_query("What is the latest stock price for Apple?")
    test_query("Tell me about Meta's performance.")

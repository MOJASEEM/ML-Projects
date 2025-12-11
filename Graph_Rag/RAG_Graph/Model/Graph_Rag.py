from neo4j import GraphDatabase
import os
from google import genai 
from google.genai.errors import APIError 
from dotenv import load_dotenv
import pandas as pd

# Load environment variables (but avoid side-effects on import)
load_dotenv()

# Default: don't perform heavy I/O or network ops at import time.
# If the caller wants to load data, they should call `load_initial_data`.

# Helper: path to CSV used by loader if invoked manually
DEFAULT_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Dataset', 'stock_data_aug_2025.csv'))
# Create a function to generate the Cypher query for a single row
def generate_merge_cypher(row):
    """
    Generates a Cypher MERGE query string based on the values in a single row (Series).
    This function replaces the hardcoded cypher_create_query.
    """
    # Note: Assuming 'row' is passed by df.apply(..., axis=1) as corrected earlier
    ticker = row.get('Ticker')
    date = str(row.get('Date'))
    open_price = row.get('Open Price')
    close_price = row.get('Close Price')
    high_price = row.get('High Price')
    low_price = row.get('Low Price')
    volume_traded = row.get('Volume Traded')
    market_cap = row.get('Market Cap')
    pe_ratio = row.get('PE Ratio')
    dividend_yield = row.get('Dividend Yield')
    eps = row.get('EPS')
    week_high = row.get('52 Week High')
    week_low = row.get('52 Week Low')
    sector = row.get('Sector')
    
    # Use f-strings and triple quotes to build the multi-line query
    # Ensure the closing triple quotes """ and the 'query = f"""' line
    # are at the same, correct indentation level (8 spaces for the content
    # inside the string is just for readability, but the key is consistency).
    # Build a safe Cypher MERGE statement (simple formatting)
    query = f"""
MERGE (c:Company {{ticker: '{ticker}', name: '{ticker}', sector: '{sector}'}})

MERGE (m:StockMetric {{
    ticker: '{ticker}',
    date: '{date}',
    open: {open_price},
    close: {close_price},
    high: {high_price},
    low: {low_price},
    volume: {volume_traded},
    market_cap: {market_cap},
    pe_ratio: {pe_ratio},
    dividend_yield: {dividend_yield},
    eps: {eps},
    week_high: {week_high},
    week_low: {week_low}
}})

MERGE (c)-[:REPORTS_ON_DATE {{date: '{date}'}}]->(m)
"""
    return query.strip()



def load_initial_data(driver, dataframe=None, csv_path=None):
    """Load rows from a DataFrame (or CSV) into the Neo4j database.

    This function is safe to call manually; it won't run on import.
    """
    if dataframe is None:
        # Try to load from CSV path if provided, otherwise the default
        path = csv_path or DEFAULT_CSV_PATH
        dataframe = pd.read_csv(path)

    # Open a session and write each row
    with driver.session() as session:
        for _, row in dataframe.iterrows():
            query = generate_merge_cypher(row)
            try:
                session.execute_write(lambda tx, q=query: tx.run(q))
            except Exception as e:
                # Log and continue
                print(f"Failed to execute merge for row {_}: {e}")

# Neo4j driver: create driver object (but avoid heavy operations on import)
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
AUTH = (os.getenv("NEO4J_USERNAME", "neo4j"), os.getenv("NEO4J_PASSWORD", "secret"))
driver = None
try:
    driver = GraphDatabase.driver(URI, auth=AUTH)
except Exception as e:
    print(f"❌ Could not create Neo4j driver: {e}")

# 1. Initialize the Gemini Client
# Initialize the client using the environment variable
# The client will automatically pick up the GEMINI_API_KEY
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
client = None
try:
    if GEMINI_KEY:
        client = genai.Client(api_key=GEMINI_KEY)
    else:
        print("⚠️ GEMINI_API_KEY not set; LLM operations will be disabled.")
except Exception as e:
    print(f"❌ Failed to initialize Gemini Client: {e}")

# 2. Define the Query Generation Function for Gemini
def generate_cypher_query_gemini(user_query, system_prompt):
    """Uses the Gemini LLM to convert a natural language question into a Cypher query."""
    
    # Structure the prompt with the system instruction
    prompt_messages = f"User Question: {user_query}\nCypher Query:"

    try:
        if client is None:
            print("generate_cypher_query_gemini: Gemini client not initialized")
            return None
        response = client.models.generate_content(
            model='gemini-2.5-flash', # Fast, free-tier model
            contents=prompt_messages,
            config=genai.types.GenerateContentConfig(
                temperature=0.0, # Keep temperature low for deterministic query generation
                system_instruction=system_prompt # Use system_instruction for clear context
            )
        )
        
        # Extract the raw text output
        cypher_query = response.text.strip()
        
        # Models sometimes add markdown (```cypher...```). Remove it if present.
        if cypher_query.startswith("```"):
            cypher_query = cypher_query.split("```")[1].strip()
            if cypher_query.lower().startswith("cypher"):
                cypher_query = cypher_query.split('\n', 1)[-1].strip()
        
        # <--- CRITICAL FIX: Ensure correct spacing between Cypher clauses
        # Replace any newlines with a single space and strip extra whitespace
        cypher_query = cypher_query.replace('\n', ' ').strip()
        
        # Replace multiple spaces with a single space
        cypher_query = ' '.join(cypher_query.split())
        
        return cypher_query

    except APIError as e:
        #print(f"❌ Gemini API Error during query generation: {e}")
        return None
    except Exception as e:
        #print(f"❌ An unexpected error occurred: {e}")
        return None
system_prompt = """
You are a Cypher query generator for a Neo4j graph database.
Your task is to convert a user's natural language question into a single, valid Cypher query.
The graph schema includes the following node labels and relationship types:

- Node Labels: Company, StockMetric, Analyst, RatingChange
- Relationship Types: REPORTS_ON_DATE, HAS_RATING, MADE_CHANGE_ON

- Properties: 
  - Company: {ticker, name, sector} 
      - IMPORTANT: The 'name' property in the database actually contains the **Ticker Symbol** (e.g., 'AAPL', 'META', 'NVDA'). 
      - The 'ticker' property ALSO contains the Ticker Symbol.
      - **ALWAYS** convert the company name mentioned in the user's question to its stock Ticker Symbol before querying.
      - Examples: 
          - "Apple" -> Use 'AAPL'
          - "Meta" -> Use 'META'
          - "Google" -> Use 'GOOGL'
          - "Microsoft" -> Use 'MSFT'
          - "Tesla" -> Use 'TSLA'
          - "Nvidia" -> Use 'NVDA'
  - StockMetric: {date, open, close, high, low, volume, market_cap, pe_ratio, dividend_yield, eps, week_high, week_low}
  - Analyst: {name}
  - RatingChange: {date, newRating, oldRating}
  
- When asked for "latest metrics" or "current price", always find the StockMetric node connected by the REPORTS_ON_DATE relationship with the maximum 'date'.
- ALWAYS return only the Cypher query, no explanations, no markdown (```cypher), and no conversational text.

Example Query for "How is Apple performing?":
MATCH (c:Company {ticker: 'AAPL'})-[:REPORTS_ON_DATE]->(m:StockMetric) RETURN m.date, m.close, m.volume ORDER BY m.date DESC LIMIT 1
"""
# 3. Update the main workflow call
# NOTE: The example/demo code below was moved under the standard
# `if __name__ == '__main__':` guard to keep the module import-safe.
def execute_cypher_query(driver, cypher_query):
    """
    Executes a Cypher query against the Neo4j database and formats the results 
    into a simple string context for the LLM.
    """
    try:
        if driver is None:
            return "No information was retrieved because the database driver is not initialized."

        with driver.session() as session:
            result = session.run(cypher_query)
            records = list(result)
            try:
                summary = result.consume()
            except Exception:
                summary = None

        # --- Format the Results for the LLM ---
        context_lines = [f"Retrieved Context from Graph:"]
        for record in records:
            # Use mapping of keys to values where possible
            try:
                data = record.data()
                result_line = ", ".join(f"{k}: {v}" for k, v in data.items())
            except Exception:
                result_line = str(record)
            context_lines.append(result_line)

        return "\n".join(context_lines)

    except Exception as e:
        print(f"Neo4j Query Execution Error: {e}")
        return "No information was retrieved due to a database error."

# ...
# 4. Integrate this new function into your main script flow
# ...

# (Integration points are provided as functions; no work is performed at import time.)
def generate_final_answer(client, user_question, context):
    """
    Generates the final, natural language answer using the structured context 
    retrieved from the Neo4j graph.
    """
    
    # Define the system prompt for the final synthesis step
    system_prompt_synthesis = (
        "You are a professional Financial Analyst. Your task is to synthesize "
        "a concise, grounded answer to the user's question, using ONLY the "
        "provided 'Retrieved Context from Graph'. "
        "If the context is empty or does not contain the necessary facts, "
        "you MUST state: 'I could not find the specific details in the current financial data graph.' "
        "Do not use external knowledge."
    )
    
    # Construct the full prompt including the context
    full_prompt = (
        f"{system_prompt_synthesis}\n\n"
        f"--- Retrieved Context from Graph ---\n{context}\n\n"
        f"--- User Question ---\n{user_question}\n\n"
        f"Final Answer:"
    )
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1 # A slightly higher temperature is okay for creative synthesis
            )
        )
        return response.text.strip()
    
    except Exception as e:
        return f"❌ Gemini API Error during final generation: {e}"

# --- Update your main script flow (Example Integration) ---
# ... (after cypher_query is executed and retrieved_data is generated) ...

# 6. Generate the Final Answer using the RAG Context
if __name__ == '__main__':
    # Demo / CLI flow when executed directly
    # Load CSV into DataFrame for demo (safe to do when running script)
    try:
        df = pd.read_csv(DEFAULT_CSV_PATH)
        print(df.head())
    except Exception as e:
        print(f"Could not read default CSV at {DEFAULT_CSV_PATH}: {e}")

    # Optionally load data into Neo4j if driver is available
    try:
        if driver is not None:
            try:
                driver.verify_connectivity()
                print("✅ Neo4j connectivity verified.")
            except Exception as e:
                print(f"Warning: could not verify connectivity: {e}")
    except Exception:
        pass

    # Example user query flow
    import sys
    if len(sys.argv) > 1:
        user_query = sys.argv[1]
    else:
        user_query = "how did the stock price of Apple change in the last month?"
    cypher_query = generate_cypher_query_gemini(user_query, system_prompt)

    if cypher_query:
        print(f"Generated Cypher:\n{cypher_query}")
        retrieved_data = execute_cypher_query(driver, cypher_query)
        final_answer = generate_final_answer(client, user_query, retrieved_data)
        print("\n--- Final Answer ---\n", final_answer)
    else:
        print("Failed to generate Cypher query.")
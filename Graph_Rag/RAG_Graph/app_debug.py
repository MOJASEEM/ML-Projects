"""
Lightweight Flask app for testing - helps identify connection issues
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import sys

# Load environment variables (Moved up for immediate availability)
env_path = os.path.join(os.path.dirname(__file__), 'Model', '.env')
load_dotenv(dotenv_path=env_path)

# Add the Model directory to the path so Graph_Rag can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Model'))

# Load environment variables from Model directory
env_path = os.path.join(os.path.dirname(__file__), 'Model', '.env')
load_dotenv(dotenv_path=env_path)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

print("=" * 60)
print("Flask Backend Startup Diagnostics")
print("=" * 60)

# Check environment variables
print("\n1. Checking environment variables:")
neo4j_uri = os.getenv("NEO4J_URI", "NOT SET")
neo4j_user = os.getenv("NEO4J_USERNAME", "NOT SET")
gemini_key = os.getenv("GEMINI_API_KEY", "NOT SET")

print(f"   NEO4J_URI: {neo4j_uri}")
print(f"   NEO4J_USERNAME: {neo4j_user}")
print(f"   GEMINI_API_KEY: {'SET' if gemini_key != 'NOT SET' else 'NOT SET'}")

if neo4j_uri == "NOT SET" or gemini_key == "NOT SET":
    print("\n  WARNING: Some required environment variables are not set!")
    print("   Please configure RAG_Graph/Model/.env file")

print("\n2. Testing imports...")
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Model'))
    from Graph_Rag import (
        generate_cypher_query_gemini,
        execute_cypher_query,
        generate_final_answer,
        driver,
        client,
        system_prompt
    )
    print("   Successfully imported RAG functions")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
except Exception as e:
    print(f"     Initialization error (may be expected): {e}")

print("\n3. Starting Flask server...")
print("=" * 60)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Backend is running"}), 200

@app.route('/api/query', methods=['POST'])
def handle_query():
    """Main endpoint to process queries"""
    try:
        data = request.json
        user_query = data.get('query')
        
        if not user_query:
            return jsonify({"error": "Query parameter is required"}), 400
        
        print(f"\n Processing query: {user_query}")
        
        #  Generate Cypher query
        print("   → Generating Cypher query...")
cypher_query = generate_cypher_query_gemini(system_prompt, user_query)
        
        if not cypher_query:
            return jsonify({
                "cypher_query": "Error generating query",
                "final_answer": "Failed to generate Cypher query."
            }), 500
        
        print(f"   Cypher: {cypher_query[:100]}...")
        
        # Execute Cypher query
        print("   → Executing Cypher query...")
        retrieved_data = execute_cypher_query(driver, cypher_query)
        print(f"    Retrieved data: {retrieved_data[:100]}...")
        
        #  Generate final answer
        print("   → Generating final answer...")
        final_answer = generate_final_answer(client, user_query, retrieved_data)
        print(f"   Answer: {final_answer[:100]}...")
        
        return jsonify({
            "cypher_query": cypher_query,
            "final_answer": final_answer
        }), 200
        
    except Exception as e:
        error_msg = str(e)
        print(f"    Error: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": error_msg,
            "cypher_query": "Error",
            "final_answer": f"Backend error: {error_msg}"
        }), 500

if __name__ == '__main__':
    import sys
    print(f"\n Backend running on http://localhost:5000")
    print(f"📍 Test health: curl http://localhost:5000/health")
    print(f" Test query: curl -X POST http://localhost:5000/api/query")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

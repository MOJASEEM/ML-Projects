from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
import sys

# Add the Model directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Model'))

# Import the RAG functions from Graph_Rag
from Graph_Rag import (
    generate_cypher_query_gemini,
    execute_cypher_query,
    generate_final_answer,
    driver,
    client,
    system_prompt
)

# Load environment variables
load_dotenv()
# Validate required environment variables
required_vars = ["NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "GEMINI_API_KEY"]
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for API routes (restrict to localhost during development)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify the backend is running"""
    return jsonify({"status": "ok", "message": "Backend is running"}), 200

@app.route('/api/query', methods=['POST'])
def handle_query():
    # Ensure request is JSON
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    data = request.get_json()
    user_query = data.get('query')
    if not user_query:
        return jsonify({"error": "'query' field is required in JSON payload"}), 400

        

    try:
        #  Generate Cypher query from user prompt
        # This was missing in your original code and is required for the next step
        cypher_query = generate_cypher_query_gemini(user_query, system_prompt)
        
        # Execute Cypher query against Neo4j
        retrieved_data = execute_cypher_query(driver, cypher_query)
        
        # Generate final answer using retrieved data
        final_answer = generate_final_answer(client, user_query, retrieved_data)
        
        return jsonify({
            "cypher_query": cypher_query,
            "final_answer": final_answer
        }), 200    
    except Exception as e:
        print(f"Error in /api/query: {str(e)}")
        return jsonify({
            "error": str(e),
            "cypher_query": "Error",
            "final_answer": f"Backend error: {str(e)}"
        }), 500

if __name__ == '__main__':
    print("Starting Flask backend on http://localhost:5000")
    print("Make sure Neo4j is running and environment variables are set")
    app.run(debug=True, host='0.0.0.0', port=5000)

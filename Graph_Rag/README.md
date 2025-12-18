# GraphRAG Stock Market Analysis

A Retrieval-Augmented Generation (RAG) system that combines Neo4j graph database with Google Gemini LLM for financial stock analysis.

## Architecture

- **Frontend**: HTML/CSS/JavaScript UI (`UI/` folder)
- **Backend**: Flask REST API (`RAG_Graph/app.py`)
- **ML Model**: RAG Pipeline (`RAG_Graph/Model/Graph_Rag.py`)
- **Database**: Neo4j Graph Database
- **LLM**: Google Gemini 2.5 Flash

## Setup Instructions

### 1. Install Dependencies
```bash
cd d:\np
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
# Navigate to RAG_Graph directory
cd RAG_Graph

# Copy the example .env file
copy .env.example .env

# Edit .env with your credentials:
# - NEO4J_URI: Your Neo4j connection URL (default: bolt://localhost:7687)
# - NEO4J_USERNAME: Neo4j username (default: neo4j)
# - NEO4J_PASSWORD: Your Neo4j password
# - GEMINI_API_KEY: Your Google Gemini API key
```

### 3. Ensure Neo4j is Running
- Make sure your Neo4j database is running and accessible at the URI specified in .env
- Verify connectivity before starting the Flask app

### 4. Start the Backend
```bash
cd RAG_Graph
python app.py
```

The backend will start on `http://localhost:5000`

### 5. Open the Frontend
Open `UI/index.html` in your web browser or serve it using a local server:
```bash
# Using Python's built-in server
cd UI
python -m http.server 8000
# Then visit http://localhost:8000
```

## How It Works

1. **User Query**: User enters a question in the UI
2. **Cypher Generation**: Gemini LLM converts natural language to Cypher query
3. **Graph Query**: Cypher query executes against Neo4j database
4. **Context Retrieval**: Structured data is retrieved from the graph
5. **Answer Synthesis**: Gemini LLM generates a final answer using the retrieved context

## Troubleshooting

### "Failed to connect to backend or process request"
- Ensure Flask backend is running on http://localhost:5000
- Check that CORS is enabled (should be in app.py)
- Open browser console (F12) to see exact error

### "Neo4j Connection Error"
- Verify Neo4j is running
- Check NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD in .env
- Ensure database credentials are correct

### "Gemini API Error"
- Verify GEMINI_API_KEY is set in .env
- Check your API key is valid on https://aistudio.google.com
- Ensure you have quota available

### "No data returned from graph"
- Verify data was loaded into Neo4j (run `python RAG_Graph/Model/Graph_Rag.py` to load initial data)
- Check that CSV files exist in `RAG_Graph/dataset/`

## File Structure
```
d:\np\
├── UI/
│   ├── index.html          # Frontend interface
│   ├── script.js           # Frontend logic
│   └── style.css           # Styling
├── RAG_Graph/
│   ├── app.py              # Flask backend server
│   ├── Model/
│   │   └── Graph_Rag.py    # RAG pipeline implementation
│   ├── dataset/            # CSV data files
│   └── .env                # Environment configuration (create from .env.example)
├── Dataset/                # Additional dataset copies
└── requirements.txt        # Python dependencies
```

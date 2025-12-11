@echo off
REM Startup script for GraphRAG application

echo ========================================
echo GraphRAG Stock Market Analysis Startup
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo Starting Flask Backend Server
echo ========================================
echo Backend will run on: http://localhost:5000
echo.
echo Make sure:
echo 1. Neo4j database is running
echo 2. Environment variables are configured in RAG_Graph\.env
echo 3. GEMINI_API_KEY is set
echo.

cd RAG_Graph
python app.py

pause

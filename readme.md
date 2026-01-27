# Install dependencies
pip install -r requirements.txt

# Option 1: Run CLI interface (existing)
python main.py

# Option 2: Run API server (new)
python api_main.py

# Option 3: Run with uvicorn directly
uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload




# Health check
curl http://localhost:8000/api/v1/forecast/health

# System info
curl http://localhost:8000/api/v1/forecast/info

# Query
curl -X POST http://localhost:8000/api/v1/forecast/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Show me actual demand for January 2025"}'
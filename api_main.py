# api_main.py
"""
FastAPI server for Load Forecasting Agentic System
Exposes REST API endpoints for frontend integration
"""

from fastapi import FastAPI, HTTPException, APIRouter, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path
import base64
import io
from datetime import datetime

# Import your existing modules
from graph import build_graph
from state import GraphState
from tools import cleanup_old_plots
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Load Forecasting Agentic System API",
    description="AI-powered load forecasting with natural language queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create router
router = APIRouter(prefix="/api/v1/forecast")

# Build graph once at startup
workflow_graph = None

@app.on_event("startup")
async def startup_event():
    """Initialize graph on startup"""
    global workflow_graph
    logger.info("[STARTUP] Building workflow graph...")
    workflow_graph = build_graph()
    logger.info("[STARTUP] ✅ Graph built successfully")
    
    # Clean up old plots
    logger.info("[STARTUP] Cleaning up old plot files...")
    cleanup_old_plots(days_to_keep=7)
    logger.info("[STARTUP] ✅ Cleanup complete")

class ChartDataset(BaseModel):
    """Chart dataset model"""
    label: str
    data: List[float]
    borderColor: str
    backgroundColor: str
    tension: Optional[float] = 0.4
    fill: Optional[bool] = False


class ChartData(BaseModel):
    """Chart data model for frontend rendering"""
    chart_type: str = Field(..., description="Type of chart: line, bar, area")
    labels: List[str] = Field(..., description="X-axis labels")
    datasets: List[ChartDataset] = Field(..., description="Chart datasets")
    title: str = Field(..., description="Chart title")
    x_axis_label: Optional[str] = Field("", description="X-axis label")
    y_axis_label: Optional[str] = Field("Value", description="Y-axis label")
    data_points: int = Field(..., description="Total number of data points")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

# -------------------------
# REQUEST & RESPONSE MODELS
# -------------------------

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    prompt: str = Field(..., description="Natural language query", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Show me actual demand for January 2025"
            }
        }


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    success: bool = Field(..., description="Whether query was successful")
    answer: str = Field(..., description="Human-readable answer")
    intent: Optional[str] = Field(None, description="Detected query intent")
    
    # Data fields
    sql_query: Optional[str] = Field(None, description="Generated SQL query")
    row_count: Optional[int] = Field(None, description="Number of rows returned")
    sample_data: Optional[List[Dict[str, Any]]] = Field(None, description="Sample data rows")
    
    # NEW: Chart data (JSON format for frontend rendering)
    has_chart: bool = Field(default=False, description="Whether chart data is available")
    chart_data: Optional[ChartData] = Field(None, description="Structured chart data for rendering")
    
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "answer": "Found 36,288 records for actual demand in January 2025",
                "intent": "data",
                "sql_query": "SELECT * FROM t_actual_demand WHERE date >= '2025-01-01'",
                "row_count": 36288,
                "sample_data": [{"date": "2025-01-01", "demand": 1250.5}],
                "has_graph": True,
                "graph_url": "/api/v1/forecast/graph/data_visualization_20260127_120000",
                "metadata": {"execution_time": "2.5s"}
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    database: str
    llm_provider: str
    version: str


class SystemInfoResponse(BaseModel):
    """System information response"""
    capabilities: List[str]
    database_type: str
    llm_provider: str
    available_tables: List[str]
    sample_queries: List[str]


# -------------------------
# ENDPOINTS
# -------------------------

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """
    Main query endpoint - processes natural language queries
    
    Handles:
    - Historical data queries
    - Demand forecasting
    - Business intelligence
    - Data visualization
    """
    logger.info(f"[API] Query received: {req.prompt[:100]}...")
    
    try:
        # Create initial state
        initial_state = GraphState(user_query=req.prompt)
        
        # Execute workflow
        logger.info("[API] Executing workflow...")
        result = workflow_graph.invoke(initial_state)
        
        # Extract results
        final_answer = result.get("final_answer", "No response generated")
        intent = result.get("intent", "unknown")
        data_ref = result.get("data_ref", {})
        graph_data = result.get("graph_data", {})
        
        # Extract SQL and data
        sql_query = None
        row_count = None
        sample_data = None
        
        if data_ref and isinstance(data_ref, dict):
            sql_query = data_ref.get("sql")
            row_count = data_ref.get("row_count", 0)
            rows = data_ref.get("rows", [])
            
            # Convert first 5 rows to dictionaries for frontend
            if rows:
                sample_data = [dict(row) if hasattr(row, 'keys') else row for row in rows[:5]]
        
        # Extract chart data (NEW FORMAT)
        has_chart = False
        chart_data_obj = None
        
        if graph_data and isinstance(graph_data, dict) and graph_data.get("ok"):
            has_chart = True
            
            # Build ChartData object
            try:
                datasets = []
                for ds in graph_data.get("datasets", []):
                    datasets.append(ChartDataset(**ds))
                
                chart_data_obj = ChartData(
                    chart_type=graph_data.get("chart_type", "line"),
                    labels=graph_data.get("labels", []),
                    datasets=datasets,
                    title=graph_data.get("title", "Chart"),
                    x_axis_label=graph_data.get("x_axis_label", ""),
                    y_axis_label=graph_data.get("y_axis_label", "Value"),
                    data_points=graph_data.get("data_points", 0),
                    metadata=graph_data.get("metadata", {})
                )
                
                logger.info(f"[API] ✅ Chart data prepared: {chart_data_obj.chart_type} with {chart_data_obj.data_points} points")
                
            except Exception as e:
                logger.error(f"[API] Error building chart data: {e}")
                has_chart = False    

        logger.info(f"[API] ✅ Query processed successfully - Intent: {intent}")
        
        return QueryResponse(
            success=True,
            answer=final_answer,
            intent=intent,
            sql_query=sql_query,
            row_count=row_count,
            sample_data=sample_data,
            has_chart=has_chart,
            chart_data=chart_data_obj,
            metadata={
                "need_db_call": result.get("need_db_call", False),
                "need_graph": result.get("need_graph", False),
                "is_out_of_range": result.get("is_out_of_range", False)
            },
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"[API] ❌ Query failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return QueryResponse(
            success=False,
            answer="An error occurred while processing your query.",
            error=str(e)
        )
        
async def health_check():
    """Health check endpoint"""
    db_type = os.getenv("DB_TYPE", "sqlite")
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    
    return HealthResponse(
        status="ok",
        database=db_type,
        llm_provider=llm_provider,
        version="1.0.0"
    )


@router.get("/info", response_model=SystemInfoResponse)
async def system_info():
    """Get system information and capabilities"""
    db_type = os.getenv("DB_TYPE", "sqlite")
    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    
    capabilities = [
        "Historical data queries",
        "Demand forecasting",
        "Business intelligence",
        "Data visualization",
        "Holiday information",
        "Performance metrics",
        "Trend analysis"
    ]
    
    available_tables = [
        "t_actual_demand",
        "t_forecasted_demand",
        "t_holidays",
        "t_metrics",
        "t_actual_weather",
        "t_forecasted_weather"
    ]
    
    sample_queries = [
        "Show me actual demand for January 2025",
        "What's the forecast for 2025-03-15?",
        "Show me a trend of actual demand",
        "Give me statistics on demand for last week",
        "What holidays are in February 2025?",
        "Compare actual vs forecasted demand"
    ]
    
    return SystemInfoResponse(
        capabilities=capabilities,
        database_type=db_type,
        llm_provider=llm_provider,
        available_tables=available_tables,
        sample_queries=sample_queries
    )


# -------------------------
# INCLUDE ROUTER
# -------------------------

app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Load Forecasting Agentic System API",
        "docs": "/docs",
        "health": "/api/v1/forecast/health",
        "info": "/api/v1/forecast/info"
    }


# -------------------------
# RUN SERVER
# -------------------------

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"[SERVER] Starting on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
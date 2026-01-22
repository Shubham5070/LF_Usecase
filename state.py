from pydantic import BaseModel
from typing import Optional, Any


class GraphState(BaseModel):
    """
    State object for the agentic workflow graph.
    Tracks all information as it flows through the agents.
    """
    
    # User input
    user_query: str

    # Query Analysis & Planning outputs
    intent: Optional[str] = None  # data | forecast | decision | text

    # Tool/Resource flags (set by planner)
    need_db_call: bool = False      # Database query required
    need_graph: bool = False        # Graph plotting required
    need_model_run: bool = False    # ML model training/execution required
    need_api_call: bool = False     # External API call required

    # Data & Execution results
    data_ref: Optional[Any] = None  # Reference to temp table or query results
    sql_query: Optional[str] = None # Generated SQL query
    
    # Model outputs (if applicable)
    model_results: Optional[Any] = None
    
    # API results (if applicable)
    api_results: Optional[Any] = None

    # Final outputs
    final_answer: Optional[str] = None
    graph_data: Optional[Any] = None  # Data prepared for plotting

    # Validation flags
    is_out_of_range: bool = False
    error_message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
# tools.py (UPDATED with database abstraction)
import requests
from langchain_core.tools import tool
import uuid
import calendar
import re
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from pathlib import Path
import time
import os
from dotenv import load_dotenv

# Import database factory
from db_factory import DatabaseFactory

# Load environment variables
load_dotenv()

# -------------------------
# CONFIGURATION
# -------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")

# Database configuration (now using factory)
DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()

PLOTS_DIR = Path("./plots")
PLOTS_DIR.mkdir(exist_ok=True)


# -------------------------
# DATABASE HELPER FUNCTIONS
# -------------------------

def get_cursor_factory():
    """Get appropriate cursor factory based on DB type"""
    if DB_TYPE == "postgresql":
        from psycopg2.extras import RealDictCursor
        return RealDictCursor
    else:  # sqlite
        # SQLite doesn't use cursor factory, we'll handle dict conversion manually
        return None


def rows_to_dict(cursor, rows):
    """Convert rows to dictionary format (works for both PostgreSQL and SQLite)"""
    if DB_TYPE == "postgresql":
        # PostgreSQL with RealDictCursor already returns dicts
        return rows
    else:  # sqlite
        # SQLite: manually convert to dict
        columns = [column[0] for column in cursor.description]
        return [dict(zip(columns, row)) for row in rows]


def adapt_sql_for_db(sql: str) -> str:
    """
    Adapt SQL syntax for the current database type
    
    PostgreSQL → SQLite conversions:
    - lf.table_name → table_name (remove schema)
    - TIMESTAMP → TEXT
    - NOW() → datetime('now')
    - DATE_TRUNC() → date() or strftime()
    """
    if DB_TYPE == "sqlite":
        # Remove schema prefix (lf.)
        sql = re.sub(r'\blf\.', '', sql)
        
        # Convert DATE_TRUNC to SQLite equivalent
        # DATE_TRUNC('day', date) → date(date)
        sql = re.sub(
            r"DATE_TRUNC\s*\(\s*['\"]day['\"]\s*,\s*(\w+)\s*\)",
            r"date(\1)",
            sql,
            flags=re.IGNORECASE
        )
        
        # DATE_TRUNC('month', date) → strftime('%Y-%m', date)
        sql = re.sub(
            r"DATE_TRUNC\s*\(\s*['\"]month['\"]\s*,\s*(\w+)\s*\)",
            r"strftime('%Y-%m', \1)",
            sql,
            flags=re.IGNORECASE
        )
        
        # DATE_TRUNC('week', date) → strftime('%Y-%W', date)
        sql = re.sub(
            r"DATE_TRUNC\s*\(\s*['\"]week['\"]\s*,\s*(\w+)\s*\)",
            r"strftime('%Y-%W', \1)",
            sql,
            flags=re.IGNORECASE
        )
        
        # Convert EXTRACT to SQLite strftime
        sql = re.sub(
            r"EXTRACT\s*\(\s*YEAR\s+FROM\s+(\w+)\s*\)",
            r"CAST(strftime('%Y', \1) AS INTEGER)",
            sql,
            flags=re.IGNORECASE
        )
        
        sql = re.sub(
            r"EXTRACT\s*\(\s*MONTH\s+FROM\s+(\w+)\s*\)",
            r"CAST(strftime('%m', \1) AS INTEGER)",
            sql,
            flags=re.IGNORECASE
        )
        
        sql = re.sub(
            r"EXTRACT\s*\(\s*DAY\s+FROM\s+(\w+)\s*\)",
            r"CAST(strftime('%d', \1) AS INTEGER)",
            sql,
            flags=re.IGNORECASE
        )
        
    return sql


# -------------------------
# CORE DB EXECUTION
# -------------------------

def execute_query(sql: str, limit: int = None) -> dict:
    """
    Executes SQL query and returns results.
    Works with both PostgreSQL and SQLite.
    
    Args:
        sql: SQL query to execute
        limit: Optional limit for number of rows to fetch
        
    Returns:
        dict with rows, row_count, and sample_rows
    """
    # Adapt SQL for current database
    sql = adapt_sql_for_db(sql)
    
    conn = DatabaseFactory.get_connection()
    
    try:
        if DB_TYPE == "postgresql":
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
        else:  # sqlite
            conn.row_factory = lambda c, r: dict(zip([col[0] for col in c.description], r))
            cursor = conn.cursor()
        
        # Execute query
        cursor.execute(sql)
        
        # Fetch results
        if limit:
            rows = cursor.fetchmany(limit)
            # Get total count
            count_sql = f"SELECT COUNT(*) as count FROM ({sql.replace(';', '')}) as subquery"
            cursor.execute(count_sql)
            count_result = cursor.fetchone()
            row_count = count_result['count'] if isinstance(count_result, dict) else count_result[0]
        else:
            rows = cursor.fetchall()
            row_count = len(rows)
        
        # Get first 3 rows for metadata/type detection
        sample_rows = rows[:3] if rows else []
        
        cursor.close()
        
        return {
            "ok": True,
            "rows": rows,
            "row_count": row_count,
            "sample_rows": sample_rows,
            "sql": sql
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "sql": sql
        }
    finally:
        conn.close()


# -------------------------
# SQL GENERATION (UPDATED)
# -------------------------

def generate_sql(user_prompt: str) -> str:
    """
    Uses LLM to generate SQL from natural language.
    Automatically adapts to PostgreSQL or SQLite based on DB_TYPE.
    """
    
    # Choose schema based on database type
    if DB_TYPE == "postgresql":
        schema_prefix = "lf."
        date_functions = """
- For date filtering: date >= 'YYYY-MM-DD' AND date < 'YYYY-MM-DD'
- For aggregations: DATE_TRUNC('day', date), DATE_TRUNC('month', date)
- For extraction: EXTRACT(YEAR FROM date), EXTRACT(MONTH FROM date)
"""
    else:  # sqlite
        schema_prefix = ""
        date_functions = """
- For date filtering: date >= 'YYYY-MM-DD' AND date < 'YYYY-MM-DD'
- For aggregations: strftime('%Y-%m-%d', date), strftime('%Y-%m', date)
- For extraction: CAST(strftime('%Y', date) AS INTEGER)
"""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": f"""
You are a {DB_TYPE.upper()} expert. Generate SIMPLE {DB_TYPE.upper()} SELECT queries ONLY.

CRITICAL RULES:
1. KEEP IT SIMPLE - No JOINs unless absolutely necessary
2. NO window functions (OVER, PARTITION BY) {'unless necessary' if DB_TYPE == 'postgresql' else ''}
3. NO subqueries unless required
{date_functions}
5. Output ONLY the SQL query, NO explanations

Database Tables (schema prefix: '{schema_prefix}'):
- {schema_prefix}t_actual_demand(datetime TEXT, date TEXT, block INTEGER, demand REAL, entrydatetime TEXT)
- {schema_prefix}t_forecasted_demand(datetime TEXT, date TEXT, block INTEGER, forecasted_demand REAL, model_id TEXT, entrydatetime TEXT)
- {schema_prefix}t_holidays(date TEXT, name TEXT, normal_holiday INTEGER, special_day INTEGER, entrydatetime TEXT)
- {schema_prefix}t_metrics(date TEXT, mape REAL, rmse REAL, model_id TEXT, entrydatetime TEXT)

EXAMPLE QUERIES:

User: "Show me actual demand for January 2025"
SQL: SELECT datetime, date, block, demand FROM {schema_prefix}t_actual_demand WHERE date >= '2025-01-01' AND date < '2025-02-01' ORDER BY datetime;

User: "Show me a trend of actual demand for January 2025"
SQL: SELECT datetime, date, block, demand FROM {schema_prefix}t_actual_demand WHERE date >= '2025-01-01' AND date < '2025-02-01' ORDER BY datetime;

User: "Get forecast for 2025-03-15"
SQL: SELECT datetime, date, block, forecasted_demand FROM {schema_prefix}t_forecasted_demand WHERE date = '2025-03-15' ORDER BY block;

User: "Show holidays in January 2025"
SQL: SELECT date, name FROM {schema_prefix}t_holidays WHERE date >= '2025-01-01' AND date < '2025-02-01' ORDER BY date;

User request: {user_prompt}

Generate SIMPLE {DB_TYPE.upper()} SQL:
""",
        "stream": False
    }

    res = requests.post(OLLAMA_URL, json=payload)
    res.raise_for_status()
    sql = res.json()["response"].strip()
    
    print(f"[SQL_GEN] Generated SQL for {DB_TYPE.upper()}: {sql}")
    
    return sql


def validate_sql(sql: str) -> str:
    """
    Validates and sanitizes SQL queries.
    """
    forbidden = ["drop", "truncate", "delete", "alter", "insert", "update"]

    if any(f in sql.lower() for f in forbidden):
        raise ValueError("Destructive SQL blocked")

    sql = " ".join(sql.strip().split())
    sql = sql.replace(";", "")

    if not sql.upper().startswith("SELECT"):
        raise ValueError("Only SELECT allowed")

    return sql + ";"


def normalize_invalid_dates(sql: str) -> str:
    """
    Fix invalid dates in SQL (e.g., 2024-02-30 → 2024-02-29).
    """
    pattern = r"'(\d{4})-(\d{2})-(\d{2})'"

    def fix(m):
        y, mth, d = map(int, m.groups())
        last = calendar.monthrange(y, mth)[1]
        return f"'{y}-{mth:02d}-{min(d,last):02d}'"

    return re.sub(pattern, fix, sql)


def strip_markdown(sql: str) -> str:
    """
    Remove markdown formatting from SQL.
    """
    return sql.replace("```sql", "").replace("```", "").replace("`", "").strip()


# -------------------------
# MAIN DB TOOL
# -------------------------

@tool
def nl_to_sql_db_tool(user_request: str) -> dict:
    """
    Convert natural language to SQL and execute it.
    Automatically works with PostgreSQL or SQLite based on DB_TYPE setting.
    
    Returns:
        - sql: The executed SQL query
        - rows: First 10 rows for preview
        - row_count: Total number of rows
        - sample_rows: First 3 rows for metadata detection
    """
    try:
        # Generate SQL
        sql = generate_sql(user_request)
        sql = strip_markdown(sql)
        sql = sql.replace('"', "'")
        
        # Validate and clean
        safe_sql = validate_sql(sql)
        safe_sql = normalize_invalid_dates(safe_sql)
        
        print(f"[NL2SQL] Final SQL ({DB_TYPE.upper()}): {safe_sql}")

        # Execute and get results (limit to 10 for preview)
        result = execute_query(safe_sql, limit=10)
        
        if not result.get("ok"):
            return {
                "ok": False,
                "sql": safe_sql,
                "error": result.get("error")
            }

        return {
            "ok": True,
            "sql": safe_sql,
            "rows": result["rows"],  # First 10 rows
            "row_count": result["row_count"],  # Total count
            "sample_rows": result["sample_rows"],  # First 3 for metadata
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "ok": False,
            "sql": sql if "sql" in locals() else None,
            "error": str(e)
        }


# -------------------------
# HELPER: Execute Aggregation Query
# -------------------------

def execute_aggregation_query(original_sql: str) -> dict:
    """
    Executes an aggregation query for large datasets.
    """
    try:
        result = execute_query(original_sql)
        return result
    except Exception as e:
        return {
            "ok": False,
            "error": str(e)
        }


# -------------------------
# GRAPH PLOTTING TOOL
# -------------------------

@tool
def graph_plotting_tool(
    sql: str,
    x_column: str, 
    y_column: str,
    plot_type: str = "line",
    title: str = "Data Visualization",
    limit: int = 10000
) -> dict:
    """
    Creates visualizations by re-executing the SQL query to fetch full data.
    Works with both PostgreSQL and SQLite.
    
    Args:
        sql: SQL query to fetch data for plotting
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        plot_type: Type of plot (line, bar, scatter)
        title: Plot title
        limit: Maximum number of data points to plot
    
    Returns:
        Plot metadata including saved file path
    """
    try:
        print(f"[GRAPH_PLOTTING] Creating {plot_type} plot using {DB_TYPE.upper()}")
        
        # Adapt SQL for current database
        sql = adapt_sql_for_db(sql)
        
        # Add limit to SQL if not already present
        if "limit" not in sql.lower():
            sql_with_limit = sql.replace(";", f" LIMIT {limit};")
        else:
            sql_with_limit = sql
        
        # Execute query to get all data for plotting
        print(f"[GRAPH_PLOTTING] Fetching data from {DB_TYPE.upper()} database")
        result = execute_query(sql_with_limit)
        
        if not result.get("ok"):
            raise ValueError(f"Query failed: {result.get('error')}")
        
        data = result["rows"]
        
        if not data:
            raise ValueError("No data returned from query")
        
        # Extract columns
        x_data = [row.get(x_column) for row in data if x_column in row]
        y_data = [row.get(y_column) for row in data if y_column in row]
        
        if not x_data or not y_data:
            raise ValueError(f"Columns {x_column} or {y_column} not found in data")
        
        print(f"[GRAPH_PLOTTING] Plotting {len(x_data)} data points")
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.style.use('seaborn-v0_8-darkgrid')
        
        if plot_type == "line":
            plt.plot(x_data, y_data, marker='o', linewidth=2, markersize=4)
        elif plot_type == "bar":
            plt.bar(range(len(x_data)), y_data, color='steelblue', alpha=0.8)
            plt.xticks(range(len(x_data)), x_data)
        elif plot_type == "scatter":
            plt.scatter(x_data, y_data, alpha=0.6, s=50)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
        
        plt.xlabel(x_column, fontsize=12, fontweight='bold')
        plt.ylabel(y_column, fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() else "_" for c in title)
        filename = f"{safe_title}_{timestamp}.png"
        filepath = PLOTS_DIR / filename
        
        # Save to file
        plt.savefig(filepath, format='png', dpi=150, bbox_inches='tight')
        print(f"[GRAPH_PLOTTING] Plot saved to {filepath}")
        
        # Also save to buffer for base64 encoding (optional)
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        plt.close()
        
        return {
            "ok": True,
            "plot_type": plot_type,
            "filepath": str(filepath),
            "filename": filename,
            "data_points": len(x_data),
            "image_base64": image_base64,
        }
        
    except Exception as e:
        print(f"[GRAPH_PLOTTING] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(e),
        }


def cleanup_old_plots(days_to_keep: int = 7):
    """
    Removes plot files older than specified days.
    """
    if not PLOTS_DIR.exists():
        return
    
    current_time = time.time()
    cutoff_time = current_time - (days_to_keep * 86400)
    
    removed_count = 0
    for plot_file in PLOTS_DIR.glob("*.png"):
        if plot_file.stat().st_mtime < cutoff_time:
            try:
                plot_file.unlink()
                removed_count += 1
            except Exception as e:
                print(f"[CLEANUP] Failed to remove {plot_file}: {e}")
    
    if removed_count > 0:
        print(f"[CLEANUP] Removed {removed_count} old plot files")


# -------------------------
# PLACEHOLDER TOOLS
# -------------------------

@tool
def model_run_tool(model_type: str = "prophet", parameters: Dict[str, Any] = None) -> dict:
    """Executes machine learning model training or prediction."""
    return {
        "ok": True,
        "model_type": model_type,
        "status": "not_implemented",
        "message": "Model execution not yet implemented"
    }


@tool
def live_api_call_tool(api_type: str, endpoint: str = None, params: Dict[str, Any] = None) -> dict:
    """Makes calls to external APIs for real-time data."""
    return {
        "ok": True,
        "api_type": api_type,
        "status": "not_implemented",
        "message": "API integration not yet implemented"
    }
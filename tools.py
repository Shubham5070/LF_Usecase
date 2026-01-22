import requests
import psycopg2
from langchain_core.tools import tool
import uuid
import calendar
import re
from psycopg2.extras import RealDictCursor
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from pathlib import Path
import time


# -------------------------
# CONFIGURATION
# -------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "load_forecasting",
    "user": "postgres",
    "password": "Test@123"
}

PLOTS_DIR = Path("./plots")
PLOTS_DIR.mkdir(exist_ok=True)


# -------------------------
# TOOL 1: DB (Database Tool) - POSTGRESQL FIXED
# -------------------------

def generate_sql(user_prompt: str) -> str:
    """
    Uses LLM to generate PostgreSQL-compatible SQL from natural language.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": f"""
You are a PostgreSQL expert. Generate ONLY PostgreSQL-compatible SQL queries.

CRITICAL POSTGRESQL RULES:
- Use PostgreSQL date functions ONLY:
  * DATE_TRUNC('day', date) for day grouping
  * DATE_TRUNC('month', date) for month grouping  
  * EXTRACT(YEAR FROM date) for extracting year
  * EXTRACT(MONTH FROM date) for extracting month
  * date >= '2025-01-01' AND date < '2025-02-01' for date ranges
- NEVER use STRFTIME (SQLite function - not supported in PostgreSQL)
- NEVER use GROUP BY with STRFTIME
- Use proper PostgreSQL column types with casts if needed
- End SQL with semicolon
- Output ONLY the SQL query, no explanations

Database Schema (PostgreSQL):
lf.t_holidays(date DATE, name TEXT, normal_holiday BOOLEAN, special_day BOOLEAN, entrydatetime TIMESTAMP)
lf.t_metrics(date DATE, mape FLOAT, rmse FLOAT, model_id TEXT, entrydatetime TIMESTAMP)
lf.t_forecasted_demand(datetime TIMESTAMP, date DATE, block INT, forecasted_demand FLOAT, model_id TEXT, entrydatetime TIMESTAMP)
lf.t_actual_demand(datetime TIMESTAMP, date DATE, block INT, demand FLOAT, entrydatetime TIMESTAMP)

PostgreSQL Date Function Examples:
✓ CORRECT: SELECT * FROM lf.t_actual_demand WHERE date >= '2025-01-01' AND date < '2025-02-01'
✓ CORRECT: SELECT DATE_TRUNC('day', date) as day, AVG(demand) FROM lf.t_actual_demand GROUP BY DATE_TRUNC('day', date)
✓ CORRECT: SELECT EXTRACT(MONTH FROM date) as month, COUNT(*) FROM lf.t_actual_demand GROUP BY EXTRACT(MONTH FROM date)
✗ WRONG: WHERE STRFTIME('%Y-%m', date) = '2025-01'  (This is SQLite, not PostgreSQL!)

User request:
{user_prompt}

PostgreSQL SQL Query:
""",
        "stream": False
    }

    res = requests.post(OLLAMA_URL, json=payload)
    res.raise_for_status()
    sql = res.json()["response"].strip()
    
    # Additional validation: detect and fix STRFTIME if LLM still uses it
    if "STRFTIME" in sql.upper() or "strftime" in sql:
        print("[SQL_GEN] Warning: Detected SQLite syntax, attempting to fix...")
        sql = fix_sqlite_to_postgresql(sql)
    
    return sql


def fix_sqlite_to_postgresql(sql: str) -> str:
    """
    Attempts to convert SQLite STRFTIME to PostgreSQL equivalents.
    """
    # Common STRFTIME patterns to PostgreSQL
    replacements = {
        r"STRFTIME\s*\(\s*['\"]%Y-%m['\"]\s*,\s*(\w+)\s*\)": r"TO_CHAR(\1, 'YYYY-MM')",
        r"STRFTIME\s*\(\s*['\"]%Y['\"]\s*,\s*(\w+)\s*\)": r"EXTRACT(YEAR FROM \1)::text",
        r"STRFTIME\s*\(\s*['\"]%m['\"]\s*,\s*(\w+)\s*\)": r"EXTRACT(MONTH FROM \1)::text",
        r"STRFTIME\s*\(\s*['\"]%W['\"]\s*,\s*(\w+)\s*\)": r"EXTRACT(WEEK FROM \1)::text",
        r"STRFTIME\s*\(\s*['\"]%d['\"]\s*,\s*(\w+)\s*\)": r"EXTRACT(DAY FROM \1)::text",
    }
    
    fixed_sql = sql
    for pattern, replacement in replacements.items():
        fixed_sql = re.sub(pattern, replacement, fixed_sql, flags=re.IGNORECASE)
    
    print(f"[SQL_FIX] Original: {sql}")
    print(f"[SQL_FIX] Fixed: {fixed_sql}")
    
    return fixed_sql


def repair_schema_references(sql: str) -> str:
    """
    Force all table references to use lf. schema.
    """
    # FROM table → FROM lf.table
    sql = re.sub(r"\bFROM\s+(?!lf\.)(\w+)", r"FROM lf.\1", sql, flags=re.IGNORECASE)

    # JOIN table → JOIN lf.table
    sql = re.sub(r"\bJOIN\s+(?!lf\.)(\w+)", r"JOIN lf.\1", sql, flags=re.IGNORECASE)

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

    if "limit" not in sql.lower():
        sql += " LIMIT 10000"  # Increased limit for larger datasets

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


def materialize_to_temp_table(select_sql: str) -> dict:
    """
    Executes SQL and stores results in a temporary table.
    """
    table = f"tmp_query_{uuid.uuid4().hex[:8]}"

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Create temp table
            cur.execute(f"CREATE TEMP TABLE {table} AS {select_sql}")

            # Count rows
            cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
            row_count = cur.fetchone()["count"]

            # Fetch preview rows
            cur.execute(f"SELECT * FROM {table} LIMIT 10")
            preview_rows = cur.fetchall()

            conn.commit()
    finally:
        conn.close()

    return {
        "type": "table",
        "name": table,
        "row_count": row_count,
        "rows": preview_rows,
    }


@tool
def nl_to_sql_db_tool(user_request: str) -> dict:
    """
    Convert natural language to SQL, execute it safely,
    and return either data or a structured execution error.
    
    This is the main DB tool (blue node in flowchart).
    """
    try:
        sql = generate_sql(user_request)
        sql = strip_markdown(sql)
        sql = sql.replace('"', "'")
        safe_sql = validate_sql(sql)
        safe_sql = repair_schema_references(safe_sql)
        safe_sql = normalize_invalid_dates(safe_sql)

        data_ref = materialize_to_temp_table(safe_sql)

        return {
            "ok": True,
            "sql": safe_sql,
            "data_ref": data_ref
        }

    except Exception as e:
        return {
            "ok": False,
            "sql": sql if "sql" in locals() else None,
            "error": str(e)
        }


# -------------------------
# HELPER FUNCTIONS
# -------------------------

def execute_db_query(sql_query: str) -> dict:
    """
    Direct database execution without NL conversion.
    Useful when SQL is already generated.
    """
    try:
        safe_sql = validate_sql(sql_query)
        safe_sql = repair_schema_references(safe_sql)
        safe_sql = normalize_invalid_dates(safe_sql)
        
        data_ref = materialize_to_temp_table(safe_sql)
        
        return {
            "ok": True,
            "sql": safe_sql,
            "data_ref": data_ref
        }
        
    except Exception as e:
        return {
            "ok": False,
            "sql": sql_query,
            "error": str(e)
        }


def fetch_data_from_temp_table(table_name: str, limit: int = 100) -> List[Dict]:
    """
    Fetches data from a temporary table created by DB tool.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            rows = cur.fetchall()
            return rows
    finally:
        conn.close()


def cleanup_temp_table(table_name: str) -> bool:
    """
    Cleans up temporary tables after use.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"[CLEANUP] Failed to drop table {table_name}: {e}")
        return False


# -------------------------
# TOOL 2: Graph Plotting
# -------------------------

@tool
def graph_plotting_tool(
    table_name: str,
    x_column: str, 
    y_column: str,
    plot_type: str = "line",
    title: str = "Data Visualization",
    limit: int = 10000
) -> dict:
    """
    Creates visualizations from database temp table.
    Fetches full data from temp table, creates plot, and saves to file.
    
    Args:
        table_name: Name of temp table containing data
        x_column: Column name for x-axis
        y_column: Column name for y-axis
        plot_type: Type of plot (line, bar, scatter)
        title: Plot title
        limit: Maximum number of data points to plot
    
    Returns:
        Plot metadata including saved file path
    """
    try:
        print(f"[GRAPH_PLOTTING] Creating {plot_type} plot from {table_name}")
        
        # Fetch full data from temp table
        print(f"[GRAPH_PLOTTING] Fetching up to {limit} rows from {table_name}")
        data = fetch_data_from_temp_table(table_name, limit=limit)
        
        if not data:
            raise ValueError(f"No data found in table {table_name}")
        
        # Extract columns
        x_data = [row.get(x_column) for row in data if x_column in row]
        y_data = [row.get(y_column) for row in data if y_column in row]
        
        if not x_data or not y_data:
            raise ValueError(f"Columns {x_column} or {y_column} not found in data")
        
        print(f"[GRAPH_PLOTTING] Plotting {len(x_data)} data points")
        
        # Create plot with better styling
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
    Call this periodically to prevent disk space issues.
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
    """
    Executes machine learning model training or prediction.
    """
    try:
        print(f"[MODEL_RUN] Running {model_type} model")
        
        result = {
            "ok": True,
            "model_type": model_type,
            "status": "completed",
            "metrics": {
                "mape": 5.2,
                "rmse": 120.5,
            },
            "predictions": [],
            "model_id": f"model_{uuid.uuid4().hex[:8]}",
        }
        
        return result
        
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "model_type": model_type,
        }


@tool
def live_api_call_tool(
    api_type: str,
    endpoint: str = None,
    params: Dict[str, Any] = None
) -> dict:
    """
    Makes calls to external APIs for real-time data.
    """
    try:
        print(f"[LIVE_API] Calling {api_type} API")
        
        return {
            "ok": True,
            "api_type": api_type,
            "data": {
                "message": "API integration pending - add your API logic here"
            },
        }
        
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "api_type": api_type,
        }
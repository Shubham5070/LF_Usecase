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
# CORE DB EXECUTION (NO TEMP TABLES)
# -------------------------

def execute_query(sql: str, limit: int = None) -> dict:
    """
    Executes SQL query and returns results directly.
    No temporary tables - just execute and fetch.
    
    Args:
        sql: SQL query to execute
        limit: Optional limit for number of rows to fetch
        
    Returns:
        dict with rows, row_count, and sample_rows
    """
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Execute query
            cur.execute(sql)
            
            # Fetch results
            if limit:
                rows = cur.fetchmany(limit)
                # Get total count
                cur.execute(f"SELECT COUNT(*) as count FROM ({sql.replace(';', '')}) as subquery")
                row_count = cur.fetchone()['count']
            else:
                rows = cur.fetchall()
                row_count = len(rows)
            
            # Get first 3 rows for metadata/type detection
            sample_rows = rows[:3] if rows else []
            
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
# SQL GENERATION
# -------------------------

def generate_sql(user_prompt: str) -> str:
    """
    Uses LLM to generate PostgreSQL-compatible SQL from natural language.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": f"""
You are a PostgreSQL expert. Generate SIMPLE PostgreSQL SELECT queries ONLY.

CRITICAL RULES:
1. KEEP IT SIMPLE - No JOINs unless absolutely necessary
2. NO window functions (OVER, PARTITION BY)
3. NO subqueries unless required
4. For date filtering, use: date >= 'YYYY-MM-DD' AND date < 'YYYY-MM-DD'
5. For aggregations, use GROUP BY with DATE_TRUNC
6. Output ONLY the SQL query, NO explanations

Database Tables:
- lf.t_actual_demand(datetime TIMESTAMP, date DATE, block INT, demand FLOAT, entrydatetime TIMESTAMP)
- lf.t_forecasted_demand(datetime TIMESTAMP, date DATE, block INT, forecasted_demand FLOAT, model_id TEXT, entrydatetime TIMESTAMP)
- lf.t_holidays(date DATE, name TEXT, normal_holiday BOOLEAN, special_day BOOLEAN, entrydatetime TIMESTAMP)
- lf.t_metrics(date DATE, mape FLOAT, rmse FLOAT, model_id TEXT, entrydatetime TIMESTAMP)

EXAMPLE QUERIES:

User: "Show me actual demand for January 2025"
SQL: SELECT datetime, date, block, demand FROM lf.t_actual_demand WHERE date >= '2025-01-01' AND date < '2025-02-01' ORDER BY datetime;

User: "Show me a trend of actual demand for January 2025"
SQL: SELECT datetime, date, block, demand FROM lf.t_actual_demand WHERE date >= '2025-01-01' AND date < '2025-02-01' ORDER BY datetime;

User: "Get forecast for 2025-03-15"
SQL: SELECT datetime, date, block, forecasted_demand FROM lf.t_forecasted_demand WHERE date = '2025-03-15' ORDER BY block;

User: "Show holidays in January 2025"
SQL: SELECT date, name FROM lf.t_holidays WHERE date >= '2025-01-01' AND date < '2025-02-01' ORDER BY date;

NEVER use:
- INNER JOIN or LEFT JOIN (unless comparing actual vs forecast)
- Window functions (OVER, PARTITION BY, ROW_NUMBER)
- Complex subqueries
- STRFTIME (SQLite function)

User request: {user_prompt}

Generate SIMPLE PostgreSQL SQL:
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
    
    # Check for overly complex queries
    if "OVER" in sql.upper() or "PARTITION BY" in sql.upper():
        print("[SQL_GEN] Warning: Detected window functions, simplifying...")
        # Fallback to simple query based on keywords
        if "actual" in user_prompt.lower() and "demand" in user_prompt.lower():
            # Extract date if present
            import re
            date_match = re.search(r'(\d{4})-(\d{2})', user_prompt)
            if date_match:
                year, month = date_match.groups()
                next_month = int(month) + 1
                next_year = year if next_month <= 12 else str(int(year) + 1)
                next_month = next_month if next_month <= 12 else 1
                sql = f"SELECT datetime, date, block, demand FROM lf.t_actual_demand WHERE date >= '{year}-{month:02d}-01' AND date < '{next_year}-{next_month:02d}-01' ORDER BY datetime;"
            elif "january" in user_prompt.lower() or "jan" in user_prompt.lower():
                year_match = re.search(r'20\d{2}', user_prompt)
                year = year_match.group(0) if year_match else "2025"
                sql = f"SELECT datetime, date, block, demand FROM lf.t_actual_demand WHERE date >= '{year}-01-01' AND date < '{year}-02-01' ORDER BY datetime;"
    
    print(f"[SQL_GEN] Generated SQL: {sql}")
    
    return sql


def fix_sqlite_to_postgresql(sql: str) -> str:
    """
    Attempts to convert SQLite STRFTIME to PostgreSQL equivalents.
    """
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
    
    print(f"[SQL_FIX] Fixed: {fixed_sql}")
    
    return fixed_sql


def repair_schema_references(sql: str) -> str:
    """
    Force all table references to use lf. schema.
    Only modifies table names in FROM and JOIN clauses, not column references.
    """
    sql = re.sub(r"\bFROM\s+(?!lf\.)([a-zA-Z_][a-zA-Z0-9_]*)\b", r"FROM lf.\1", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\bJOIN\s+(?!lf\.)([a-zA-Z_][a-zA-Z0-9_]*)\b", r"JOIN lf.\1", sql, flags=re.IGNORECASE)
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
# MAIN DB TOOL (NO TEMP TABLES)
# -------------------------

@tool
def nl_to_sql_db_tool(user_request: str) -> dict:
    """
    Convert natural language to SQL and execute it.
    Returns results directly - NO temp tables created.
    
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
        safe_sql = repair_schema_references(safe_sql)
        safe_sql = normalize_invalid_dates(safe_sql)
        
        # Clean up any lf. in function arguments
        safe_sql = re.sub(r'(EXTRACT\s*\([^)]*FROM\s+)lf\.', r'\1', safe_sql, flags=re.IGNORECASE)
        safe_sql = re.sub(r'(DATE_TRUNC\s*\([^,]+,\s*)lf\.', r'\1', safe_sql, flags=re.IGNORECASE)
        
        print(f"[NL2SQL] Final SQL: {safe_sql}")

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
# GRAPH PLOTTING TOOL (DIRECT DB FETCH)
# -------------------------

@tool
def graph_plotting_tool(
    sql: str,  # Changed: now accepts SQL instead of table_name
    x_column: str, 
    y_column: str,
    plot_type: str = "line",
    title: str = "Data Visualization",
    limit: int = 10000
) -> dict:
    """
    Creates visualizations by re-executing the SQL query to fetch full data.
    
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
        print(f"[GRAPH_PLOTTING] Creating {plot_type} plot")
        
        # Add limit to SQL if not already present
        if "limit" not in sql.lower():
            sql_with_limit = sql.replace(";", f" LIMIT {limit};")
        else:
            sql_with_limit = sql
        
        # Execute query to get all data for plotting
        print(f"[GRAPH_PLOTTING] Fetching data from database")
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
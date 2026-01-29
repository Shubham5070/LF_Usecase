# tools.py
# ------------------------------------------------------
# Database + Visualization Tools (PostgreSQL / SQLite)
# ------------------------------------------------------

import os
import re
import io
import time
import base64
import calendar
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Project imports
from db_factory import DatabaseFactory
from agent_llm_config import get_agent_llm

# ✅ DATA AVAILABILITY (GROUND TRUTH)
from data_availability import (
    ACTUAL_DATA_START,
    ACTUAL_DATA_END,
    FORECAST_DATA_START,
    FORECAST_DATA_END,
    HOLIDAY_START,
    HOLIDAY_END,
    METRICS_START,
    METRICS_END,
    build_out_of_range_message,  # 👈 ADD THIS
)


# ------------------------------------------------------
# LOGICAL CURRENT TIME (HARDCODED)
# ------------------------------------------------------

LOGICAL_NOW = datetime(2025, 10, 14, 0, 0, 0)
LOGICAL_NOW_DATE = LOGICAL_NOW.date()

# ------------------------------------------------------
# ENV & CONFIGURATION
# ------------------------------------------------------

load_dotenv()

DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()

PLOTS_DIR = Path("./plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------
# DATABASE UTILITIES
# ------------------------------------------------------
def adapt_sql_for_db(sql: str) -> str:
    """
    Adapt PostgreSQL SQL syntax to SQLite when required.
    """
    if DB_TYPE != "sqlite":
        return sql

    # Remove schema prefix
    sql = re.sub(r"\blf\.", "", sql)

    # 🔒 Fix malformed DATE('now', ...) expressions (LLM-generated garbage)
    sql = re.sub(
        r"DATE\s*\(\s*'now'\s*,[^)]*\)",
        f"'{LOGICAL_NOW_DATE}'",
        sql,
        flags=re.IGNORECASE,
    )

    # DATE_TRUNC replacements
    sql = re.sub(
        r"DATE_TRUNC\s*\(\s*['\"]day['\"]\s*,\s*(\w+)\s*\)",
        r"date(\1)",
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"DATE_TRUNC\s*\(\s*['\"]month['\"]\s*,\s*(\w+)\s*\)",
        r"strftime('%Y-%m', \1)",
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"DATE_TRUNC\s*\(\s*['\"]week['\"]\s*,\s*(\w+)\s*\)",
        r"strftime('%Y-%W', \1)",
        sql,
        flags=re.IGNORECASE,
    )

    # EXTRACT replacements
    sql = re.sub(
        r"EXTRACT\s*\(\s*YEAR\s+FROM\s+(\w+)\s*\)",
        r"CAST(strftime('%Y', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"EXTRACT\s*\(\s*MONTH\s+FROM\s+(\w+)\s*\)",
        r"CAST(strftime('%m', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"EXTRACT\s*\(\s*DAY\s+FROM\s+(\w+)\s*\)",
        r"CAST(strftime('%d', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE,
    )

    # hour(datetime) → SQLite-compatible
    sql = re.sub(
        r"\bhour\s*\(\s*(\w+)\s*\)",
        r"CAST(strftime('%H', \1) AS INTEGER)",
        sql,
        flags=re.IGNORECASE,
    )

    return sql



def execute_query(sql: str, limit: int | None = None) -> dict:
    """
    Execute SQL query for PostgreSQL or SQLite.
    """
    sql = adapt_sql_for_db(sql)
    conn = DatabaseFactory.get_connection()

    try:
        if DB_TYPE == "postgresql":
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
        else:
            conn.row_factory = lambda c, r: dict(
                zip([col[0] for col in c.description], r)
            )
            cursor = conn.cursor()

        cursor.execute(sql)

        if limit:
            rows = cursor.fetchmany(limit)
            count_sql = f"SELECT COUNT(*) as count FROM ({sql.replace(';', '')}) sub"
            cursor.execute(count_sql)
            row_count = cursor.fetchone()["count"]
        else:
            rows = cursor.fetchall()
            row_count = len(rows)

        return {
            "ok": True,
            "rows": rows,
            "row_count": row_count,
            "sample_rows": rows[:3],
            "sql": sql,
        }

    except Exception as e:
        return {"ok": False, "error": str(e), "sql": sql}

    finally:
        conn.close()


# ------------------------------------------------------
# SQL GENERATION (LLM)
# ------------------------------------------------------

def generate_sql(user_prompt: str) -> str:
    """
    Generate database-specific SQL using agent-configured LLM,
    grounded in actual data availability.
    """
    llm = get_agent_llm("nl_to_sql")

    schema = "lf." if DB_TYPE == "postgresql" else ""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
You are a {DB_TYPE.upper()} SQL expert.
Generate SIMPLE SELECT queries only.

CRITICAL RULES:
- ONLY SELECT queries
- NO DELETE / UPDATE / INSERT
- Avoid JOINs unless absolutely required
- NEVER invent columns
- Output ONLY SQL
- Model performance MUST be evaluated using t_metrics only
- Also return only the SQL query, without any explanations or markdown formatting.


AVAILABLE DATA RANGES:
- Actual demand: {ACTUAL_DATA_START.date()} → {ACTUAL_DATA_END.date()}
- Forecasted demand: {FORECAST_DATA_START.date()} → {FORECAST_DATA_END.date()}
- Holidays: {HOLIDAY_START} → {HOLIDAY_END}
- Metrics: {METRICS_START} → {METRICS_END}

DATE RULES:
- "last N days" or "last month" is relative to {LOGICAL_NOW_DATE}
- For daily aggregation: GROUP BY date only

TABLES:
- {schema}t_actual_demand(datetime, date, block, demand, entrydatetime)
  → For time-series data: SELECT datetime, demand (not just date)
  → Multiple readings per day? Use datetime as X-axis
- {schema}t_forecasted_demand(datetime, date, block, forecasted_demand, model_id, entrydatetime)
  → For time-series data: SELECT datetime, forecasted_demand (not just date)
  → Multiple readings per day? Use datetime as X-axis
- {schema}t_holidays(date, name, normal_holiday, special_day, entrydatetime)
- {schema}t_metrics(date, mape, rmse, model_id, entrydatetime)

COLUMN SELECTION RULES:
- When fetching demand (actual or forecasted) and result may have >50 rows:
  → Prefer 'datetime' over 'date' for better visualization
  → Only use 'date' if explicitly asking for daily aggregates
- Always include block number if showing block-level data if block number not there then ignore it.
""",
            ),
            ("user", "{user_prompt}"),
        ]
    )

    return llm.invoke(
        prompt.format_messages(user_prompt=user_prompt)
    ).content.strip()


def strip_markdown(sql: str) -> str:
    return sql.replace("```sql", "").replace("```", "").replace("`", "").strip()


def validate_sql(sql: str) -> str:
    forbidden = ["drop", "delete", "truncate", "alter", "insert", "update"]
    if any(word in sql.lower() for word in forbidden):
        raise ValueError("Destructive SQL detected")

    sql = " ".join(sql.split()).replace(";", "")
    if not sql.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    return sql + ";"


def normalize_invalid_dates(sql: str) -> str:
    pattern = r"'(\d{4})-(\d{2})-(\d{2})'"

    def fix(match):
        y, m, d = map(int, match.groups())
        last_day = calendar.monthrange(y, m)[1]
        return f"'{y}-{m:02d}-{min(d, last_day):02d}'"

    return re.sub(pattern, fix, sql)


def generate_plot_code(sql: str, columns: list[str], sample_rows: list[dict]) -> str:
    """
    Generate matplotlib code dynamically based on SQL + result schema.
    Automatically detects the best visualization type based on data characteristics.
    """
    llm = get_agent_llm("summarization")

    from langchain_core.messages import HumanMessage, SystemMessage

    # Prepare the data strings
    columns_str = ', '.join(columns)
    sample_row_str = str(sample_rows[0]) if sample_rows else "{}"

    system_msg = SystemMessage(content="""You are a data visualization expert. Analyze the data structure and automatically select the BEST graph type.

CRITICAL RULES:
1. Use matplotlib.pyplot as plt
2. Assume `data` is a list of dicts already available
3. DO NOT read files
4. DO NOT show the plot (no plt.show())
5. MUST save to buffer with: plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
6. Output ONLY executable Python code, NO markdown, NO explanations
7. AUTOMATICALLY detect the best visualization type based on data structure

GRAPH TYPE SELECTION LOGIC:
1. **Line Plot** - Use when:
   - Time-series data (datetime/date column present)
   - Continuous numeric data over sequential values
   - Columns: datetime/date + demand/forecasted_demand/metric

2. **Bar Chart** - Use when:
   - Categorical X-axis with numeric Y values
   - Comparing discrete values across categories
   - Aggregated data by category (e.g., monthly averages, block-wise totals)
   - Columns: category/block/month + count/sum/average

3. **Scatter Plot** - Use when:
   - Two continuous numeric variables
   - Looking for correlations or patterns
   - Columns: numeric_x + numeric_y (e.g., actual vs forecasted)

4. **Multiple Lines** - Use when:
   - Comparing multiple series over time
   - Data has model_id or multiple forecast sources
   - Columns: datetime + value + category/model_id

5. **Horizontal Bar** - Use when:
   - Long category names that need space
   - Ranking or comparison of named entities
   - Columns: name/label + numeric_value

6. **Stacked/Grouped Bar** - Use when:
   - Comparing subcategories within categories
   - Data has nested grouping (e.g., block + model_id)

DATA HANDLING RULES:
- For datetime/date columns: ALWAYS use index-based X-axis for 20+ points
- For large datasets (50+ rows): Use grid, minimal labels
- For text labels: Rotate 45° if >10 labels
- Always use tight_layout()
- Use appropriate colors: blues for demand, greens for metrics, reds for errors

DETECTION EXAMPLES:

Example 1 - Time-series (datetime + demand):
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
x_values = list(range(len(data)))
y_values = [float(row['demand']) for row in data]
ax.plot(x_values, y_values, linewidth=2, color='#2E86AB', marker='o', markersize=3)
ax.set_title('Demand Over Time', fontsize=14, fontweight='bold')
ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Demand (MW)', fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
```

Example 2 - Categorical comparison (block + average):
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
labels = [str(row['block']) for row in data]
values = [float(row['avg_demand']) for row in data]
ax.bar(range(len(values)), values, color='#06A77D', alpha=0.8, edgecolor='black')
ax.set_title('Average Demand by Block', fontsize=14, fontweight='bold')
ax.set_xlabel('Block', fontsize=11)
ax.set_ylabel('Average Demand (MW)', fontsize=11)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=0)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
```

Example 3 - Comparison (actual vs forecasted):
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
actual = [float(row['demand']) for row in data]
forecast = [float(row['forecasted_demand']) for row in data]
ax.scatter(actual, forecast, alpha=0.6, s=50, color='#2E86AB', edgecolors='black', linewidth=0.5)
ax.plot([min(actual), max(actual)], [min(actual), max(actual)], 'r--', linewidth=2, label='Perfect Forecast')
ax.set_title('Actual vs Forecasted Demand', fontsize=14, fontweight='bold')
ax.set_xlabel('Actual Demand (MW)', fontsize=11)
ax.set_ylabel('Forecasted Demand (MW)', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
```

Example 4 - Multiple models comparison:
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 6))
models = {}
for row in data:
    model_id = row.get('model_id', 'unknown')
    if model_id not in models:
        models[model_id] = {'x': [], 'y': []}
    models[model_id]['x'].append(len(models[model_id]['x']))
    models[model_id]['y'].append(float(row['forecasted_demand']))
colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']
for idx, (model_id, values) in enumerate(models.items()):
    ax.plot(values['x'], values['y'], linewidth=2, marker='o', markersize=3, 
            label=f'Model {model_id}', color=colors[idx % len(colors)])
ax.set_title('Forecast Comparison by Model', fontsize=14, fontweight='bold')
ax.set_xlabel('Time Index', fontsize=11)
ax.set_ylabel('Forecasted Demand (MW)', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
```

Example 5 - Metrics visualization (mape/rmse):
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
labels = [str(row['date']) for row in data]
values = [float(row['mape']) for row in data]
colors = ['#06A77D' if v < 5 else '#F18F01' if v < 10 else '#D00000' for v in values]
ax.bar(range(len(values)), values, color=colors, alpha=0.8, edgecolor='black')
ax.axhline(y=5, color='green', linestyle='--', linewidth=1, label='Good (<5%)')
ax.axhline(y=10, color='orange', linestyle='--', linewidth=1, label='Acceptable (<10%)')
ax.set_title('MAPE by Date', fontsize=14, fontweight='bold')
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('MAPE (%)', fontsize=11)
if len(labels) > 10:
    step = max(1, len(labels) // 10)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels(labels[::step], rotation=45, ha='right')
else:
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend()
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
```

ANALYZE the columns and sample data, DETERMINE the best graph type, then generate appropriate code.""")

    user_msg = HumanMessage(content=f"""Analyze this data structure and generate the BEST visualization:

Columns: {columns_str}
Sample row: {sample_row_str}
SQL Query: {sql}

INSTRUCTIONS:
1. Identify the data type (time-series, categorical, comparison, metrics)
2. Select the most appropriate graph type
3. Generate clean, professional visualization code
4. Return ONLY executable Python code, no explanations or markdown""")

    messages = [system_msg, user_msg]
    code = llm.invoke(messages).content.strip()
    
    # Clean up code - remove markdown if present
    code = code.replace("```python", "").replace("```", "").strip()
    
    return code


# ------------------------------------------------------
# RELATIVE DATE RESOLUTION (LOGICAL NOW)
# ------------------------------------------------------

def replace_relative_dates_with_logical_now(sql: str) -> str:
    """
    Replace DATE('now', '-N days/months') with fixed logical dates.
    """

    def repl_days(match):
        days = int(match.group(1))
        return f"'{(LOGICAL_NOW_DATE - timedelta(days=days))}'"

    def repl_months(match):
        months = int(match.group(1))
        return f"'{(LOGICAL_NOW_DATE - timedelta(days=30 * months))}'"

    sql = re.sub(
        r"DATE\s*\(\s*'now'\s*,\s*'-\s*(\d+)\s*days'\s*\)",
        repl_days,
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
        r"DATE\s*\(\s*'now'\s*,\s*'-\s*(\d+)\s*month[s]?'\s*\)",
        repl_months,
        sql,
        flags=re.IGNORECASE,
    )

    sql = re.sub(
    r"\bMONTH\s*\(\s*(\w+)\s*\)",
    r"CAST(strftime('%m', \1) AS INTEGER)",
    sql,
    flags=re.IGNORECASE,
    )


    return sql


# ------------------------------------------------------
# HARD SAFETY: DATE CLAMPING
# ------------------------------------------------------

def clamp_dates_to_availability(sql: str) -> str:
    sql = re.sub(
        r"date\s*<\s*'(\d{4}-\d{2}-\d{2})'",
        f"date <= '{ACTUAL_DATA_END.date()}'",
        sql,
        flags=re.IGNORECASE,
    )
    sql = re.sub(
        r"date\s*>\s*'(\d{4}-\d{2}-\d{2})'",
        f"date >= '{ACTUAL_DATA_START.date()}'",
        sql,
        flags=re.IGNORECASE,
    )
    return sql


# ------------------------------------------------------
# MAIN NL → SQL TOOL
# ------------------------------------------------------

@tool
def nl_to_sql_db_tool(user_request: str) -> dict:
    """
    Convert natural language to SQL and execute it.
    """
    try:
        sql = generate_sql(user_request)
        sql = strip_markdown(sql).replace('"', "'")
        sql = validate_sql(sql)
        sql = normalize_invalid_dates(sql)
        sql = replace_relative_dates_with_logical_now(sql)
        sql = clamp_dates_to_availability(sql)

        result = execute_query(sql, limit=10)

        if not result["ok"]:
            return result

        # 🚫 BLOCK "before data availability" queries
        if (
            result["row_count"] > 0
            and "before" in user_request.lower()
        ):
            return {
                "ok": False,
                "message": build_out_of_range_message("actual"),
                "sql": sql,
            }

        return {
            "ok": True,
            "sql": sql,
            "rows": result["rows"],
            "row_count": result["row_count"],
            "sample_rows": result["sample_rows"],
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------
# GRAPH PLOTTING TOOL
# ------------------------------------------------------
@tool
def graph_plotting_tool(sql: str) -> dict:
    """
    Dynamically generate and execute visualization code.
    ALWAYS executes the SQL query against the database to fetch fresh data.
    """
    try:
        print(f"[GRAPH] Received SQL query for plotting")
        print(f"[GRAPH] Original SQL: {sql}")
        
        # 🔒 COMPULSORY DATABASE CALL - Always fetch fresh data
        sql = adapt_sql_for_db(sql)
        print(f"[GRAPH] Adapted SQL: {sql}")
        
        print(f"[GRAPH] Executing query against database...")
        result = execute_query(sql)
        
        if not result["ok"]:
            print(f"[GRAPH] ✗ Database query failed: {result.get('error', 'Unknown error')}")
            return {
                "ok": False,
                "error": f"Database query failed: {result.get('error', 'Unknown error')}",
                "sql": sql,
            }

        if not result["rows"]:
            print(f"[GRAPH] ✗ No data returned from database")
            return {
                "ok": False,
                "error": "No data returned from database query",
                "sql": sql,
            }

        data = result["rows"]
        row_count = len(data)
        print(f"[GRAPH] ✓ Database returned {row_count} rows")
        
        columns = list(data[0].keys())
        sample_rows = data[:5]
        print(f"[GRAPH] Columns: {columns}")

        # Generate dynamic plot code based on data structure
        print(f"[GRAPH] Generating dynamic plot code...")
        plot_code = generate_plot_code(sql, columns, sample_rows)
        
        print(f"[GRAPH] Generated plot code ({len(plot_code)} chars)")
        print(f"[GRAPH] Plot code preview:\n{plot_code[:200]}...")

        # ---- SAFE EXECUTION CONTEXT ----
        buffer = io.BytesIO()

        SAFE_GLOBALS = {
            "__builtins__": {
                "__import__": __import__,
                "len": len,
                "range": range,
                "list": list,
                "dict": dict,
                "float": float,
                "int": int,
                "str": str,
                "enumerate": enumerate,
            },
            "plt": plt,
            "io": io,
        }

        SAFE_LOCALS = {
            "data": data,
            "buffer": buffer,
        }

        print(f"[GRAPH] Executing plot code...")
        exec(plot_code, SAFE_GLOBALS, SAFE_LOCALS)
        print(f"[GRAPH] ✓ Code executed successfully")

        buffer.seek(0)
        image_data = buffer.read()
        image_base64 = base64.b64encode(image_data).decode()

        # Save plot to plots folder
        import os
        from datetime import datetime
        plots_dir = "plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(plots_dir, f"plot_{timestamp}.png")
        with open(plot_filename, "wb") as f:
            f.write(image_data)
        print(f"[GRAPH] ✓ Plot saved to {plot_filename}")

        return {
                "ok": True,
                "image_base64": image_base64,
                "plot_file": plot_filename,
                "generated_plot_code": plot_code,
                "plot_type": "dynamic",
                "rows_plotted": row_count,
                "sql_executed": sql,
            }


    except Exception as e:
        import traceback
        print(f"[GRAPH] ✗ Error: {e}")
        print(f"[GRAPH] Traceback:\n{traceback.format_exc()}")
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ------------------------------------------------------
# AGGREGATION QUERY HELPER (REQUIRED BY AGENTS)
# ------------------------------------------------------

def execute_aggregation_query(original_sql: str) -> dict:
    """
    Executes aggregation SQL generated by data_observation_agent.
    Keeps interface stable for agent imports.
    """
    try:
        return execute_query(original_sql)
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "sql": original_sql,
        }


# ------------------------------------------------------
# MAINTENANCE
# ------------------------------------------------------

def cleanup_old_plots(days_to_keep: int = 7):
    cutoff = time.time() - days_to_keep * 86400
    for file in PLOTS_DIR.glob("*.png"):
        if file.stat().st_mtime < cutoff:
            file.unlink(missing_ok=True)


# ------------------------------------------------------
# PLACEHOLDER TOOLS (KEPT AS-IS)
# ------------------------------------------------------

@tool
def model_run_tool(model_type: str = "prophet", parameters: Dict[str, Any] = None) -> dict:
    """
    Run a Demand forecasting models.
    """
    return {
        "ok": True,
        "model_type": model_type,
        "status": "not_implemented",
    }


@tool
def live_api_call_tool(
    api_type: str, endpoint: str = None, params: Dict[str, Any] = None
) -> dict:
    "call an external live API and return the results"
    return {
        "ok": True,
        "api_type": api_type,
        "status": "not_implemented",
    }
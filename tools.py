# tools.py - MERGED & CORRECTED VERSION
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
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    build_out_of_range_message,
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
    
    # MONTH() function
    sql = re.sub(
        r"\bMONTH\s*\(\s*(\w+)\s*\)",
        r"CAST(strftime('%m', \1) AS INTEGER)",
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
        print(f"[DB ERROR] {e}")
        print(f"[DB ERROR] SQL: {sql}")
        return {"ok": False, "error": str(e), "sql": sql}

    finally:
        conn.close()


# ------------------------------------------------------
# SQL GENERATION (LLM) - IMPROVED FOR GRAPHS
# ------------------------------------------------------
def generate_sql(user_prompt: str) -> str:
    """
    IMPROVED: Generate database-specific SQL with strict rules
    for single-day vs multi-day trend queries.
    """

    llm = get_agent_llm("nl_to_sql")
    schema = "lf." if DB_TYPE == "postgresql" else ""

    import re

    # --------------------------------------------------
    # 🔒 HARD OVERRIDE: SINGLE DATE → RAW DATA ONLY
    # --------------------------------------------------
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", user_prompt)

    if len(dates) == 1:
        date = dates[0]
        return f"""
        SELECT
            datetime,
            date,
            block,
            demand
        FROM {schema}t_actual_demand
        WHERE date = '{date}'
        ORDER BY datetime;
        """

    # --------------------------------------------------
    # Detect comparison query
    # --------------------------------------------------
    is_comparison = any(word in user_prompt.lower() for word in [
        'compare', 'comparison', 'vs', 'versus', 'against',
        'actual and forecast', 'forecast and actual'
    ])

    # --------------------------------------------------
    # LLM PROMPT (CLEAN + NON-CONTRADICTING)
    # --------------------------------------------------
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
You are a {DB_TYPE.upper()} SQL expert specializing in time-series demand data.

CRITICAL RULES:
- Generate ONLY SELECT queries
- NO DELETE / UPDATE / INSERT / DROP
- Output ONLY SQL (no explanation, no markdown)
- End query with semicolon

TABLES:
1. {schema}t_actual_demand(datetime, date, block, demand)
2. {schema}t_forecasted_demand(datetime, date, block, forecasted_demand, model_id)

ABSOLUTE RULES (DO NOT VIOLATE):

1️⃣ SINGLE DATE (even if word "trend" is used):
- ALWAYS return ALL 96 blocks
- NEVER use GROUP BY

Example:
SELECT datetime, date, block, demand
FROM {schema}t_actual_demand
WHERE date = '2025-05-26'
ORDER BY datetime;

2️⃣ MULTI-DATE COMPARISON (2–7 days, "compare", "vs"):
- NO GROUP BY
- Return raw block-level data

Example:
SELECT datetime, date, block, demand
FROM {schema}t_actual_demand
WHERE date IN ('2025-05-15', '2025-05-16')
ORDER BY date, datetime;

3️⃣ LONG-TERM TREND (ONLY when range ≥ 8 days OR full month):
- USE GROUP BY date
- ONE row per day

Example:
SELECT
    date,
    AVG(demand) AS avg_demand
FROM {schema}t_actual_demand
WHERE date BETWEEN '2025-03-01' AND '2025-03-31'
GROUP BY date
ORDER BY date;

🚫 NEVER use GROUP BY when:
- Date range is less than 8 days
- Only ONE date is requested
- User wants intraday / bar chart / block-wise trend

🔥 ABSOLUTE TIME FILTER RULE (NON-NEGOTIABLE):
- NEVER use strftime(), extract(), month(), or year() for filtering
- NEVER compare month numbers (e.g., BETWEEN 3 AND 5)
- ALL time logic MUST use FULL DATE comparisons:
  → date = 'YYYY-MM-DD'
  → date IN ('YYYY-MM-DD', ...)
  → date BETWEEN 'YYYY-MM-DD' AND 'YYYY-MM-DD'
- If user mentions months/seasons (e.g., "March to May", "Summer"):
  → Convert them into FULL DATE RANGES
  → DO NOT extract month or year from date column

METRICS QUERY RULES (MANDATORY):
- If user asks for MAPE / RMSE for a specific date:
  → Query ONLY t_metrics
  → Filter ONLY by date
  → DO NOT infer or derive model_id
- NEVER use subqueries, joins, LIMIT, or date ranges for metrics
- Use model_id ONLY if user explicitly asks for a specific model

AVAILABLE DATA RANGES:
- Actual demand: {ACTUAL_DATA_START.date()} → {ACTUAL_DATA_END.date()}
- Forecasted demand: {FORECAST_DATA_START.date()} → {FORECAST_DATA_END.date()}
- Holidays: {HOLIDAY_START} → {HOLIDAY_END}
- Metrics: {METRICS_START} → {METRICS_END}

TIME BLOCK MAPPING (1–96, each block = 15 minutes):
- Block 1  = 00:00–00:15  (Midnight)
- Block 24 = 05:45–06:00  (Pre-dawn)
- Block 25 = 06:00–06:15  (Early morning start)
- Block 37 = 09:00–09:15  (Morning peak start)
- Block 48 = 11:45–12:00  (Midday)
- Block 61 = 15:00–15:15  (Afternoon peak start)
- Block 72 = 17:45–18:00  (Evening peak start)
- Block 84 = 20:45–21:00  (Evening peak end)
- Block 96 = 23:45–00:00  (End of day)

NAMED BLOCK PERIODS:
- Off-Peak / Night:      blocks 1–24   (00:00–06:00)
- Morning Ramp:          blocks 25–36  (06:00–09:00)
- Morning Peak:          blocks 37–48  (09:00–12:00)
- Midday:                blocks 49–60  (12:00–15:00)
- Afternoon Peak:        blocks 61–72  (15:00–18:00)
- Evening Peak:          blocks 73–84  (18:00–21:00)
- Night Ramp-down:       blocks 85–96  (21:00–00:00)

SEASON DEFINITIONS (Indian calendar):
- Winter:  Dec, Jan, Feb
- Summer:  March, April, May
- Monsoon: June, July, Aug, Sep
- Post-Monsoon: Oct, Nov

SEASON DETECTION IN SQL (STRICT – DATE RANGE ONLY):
- Winter:
  date BETWEEN 'YYYY-12-01' AND 'YYYY-02-28'
- Summer:
  date BETWEEN 'YYYY-03-01' AND 'YYYY-05-31'
- Monsoon:
  date BETWEEN 'YYYY-06-01' AND 'YYYY-09-30'
- Post-Monsoon:
  date BETWEEN 'YYYY-10-01' AND 'YYYY-11-30'

⚠️ CRITICAL:
- DO NOT use strftime(), CAST(strftime), EXTRACT, or month logic
- DO NOT compare numeric months (e.g., BETWEEN 3 AND 5)
- ALWAYS expand seasons/months into explicit date ranges

KEY DECISION LOGIC:
- 1 date → RAW
- 2–7 dates → RAW
- 8+ dates OR full month → AGGREGATED

Current logical date: {LOGICAL_NOW_DATE}
""",
            ),
            ("user", "{user_prompt}"),
        ]
    )

    response = llm.invoke(
        prompt.format_messages(user_prompt=user_prompt)
    ).content.strip()

    print(f"[SQL_GEN] Raw LLM output: {response[:200]}...")

    return response


def strip_markdown(sql: str) -> str:
    """IMPROVED: Remove markdown formatting from SQL"""
    # Remove markdown code blocks
    sql = sql.replace("```sql", "").replace("```", "").replace("`", "").strip()
    
    # Remove any explanatory text before SELECT
    if "SELECT" in sql.upper():
        select_pos = sql.upper().find("SELECT")
        sql = sql[select_pos:]
    
    # Remove any text after final semicolon
    if ";" in sql:
        # Find last semicolon and cut everything after
        last_semicolon = sql.rfind(";")
        sql = sql[:last_semicolon + 1]
    
    return sql.strip()


def validate_sql(sql: str) -> str:
    """Validate SQL is safe and correct"""
    forbidden = ["drop", "delete", "truncate", "alter", "insert", "update"]
    if any(word in sql.lower() for word in forbidden):
        raise ValueError("Destructive SQL detected")

    sql = " ".join(sql.split()).replace(";", "")
    if not sql.upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed")

    return sql + ";"


def normalize_invalid_dates(sql: str) -> str:
    """Fix invalid dates like 2025-05-35"""
    pattern = r"'(\d{4})-(\d{2})-(\d{2})'"

    def fix(match):
        y, m, d = map(int, match.groups())
        try:
            last_day = calendar.monthrange(y, m)[1]
            return f"'{y}-{m:02d}-{min(d, last_day):02d}'"
        except:
            return f"'{y}-{m:02d}-01'"

    return re.sub(pattern, fix, sql)


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
# HELPER FUNCTIONS FOR PLOT DETECTION
# ------------------------------------------------------

def _detect_multi_date_comparison(data: list, user_query: str) -> bool:
    """
    Detect if this is a multi-date comparison query.
    E.g., "compare 15 august vs 16 august", "15 and 16 august demand"
    """
    if not data or len(data) == 0:
        return False
    
    if not isinstance(data[0], dict):
        return False
    
    comparison_keywords = ['compare', 'vs', 'versus', 'between', 'and']
    query_lower = user_query.lower()
    
    has_comparison_keyword = any(kw in query_lower for kw in comparison_keywords)
    
    # Check if 'date' column exists
    if 'date' not in data[0]:
        return False
    
    # Count unique dates
    try:
        unique_dates = set()
        for row in data:
            if not isinstance(row, dict):
                continue
            date_val = row.get('date')
            if date_val is not None:
                unique_dates.add(str(date_val))
        
        has_multiple_dates = len(unique_dates) > 1
    except Exception as e:
        print(f"[MULTI_DATE_DETECT] Error counting dates: {e}")
        return False
    
    columns = list(data[0].keys())
    has_date_col = 'date' in columns
    has_demand_col = any(col in columns for col in ['demand', 'actual_demand', 'forecasted_demand', 'predicted_demand'])
    
    result = has_comparison_keyword and has_multiple_dates and has_date_col and has_demand_col
    
    print(f"[MULTI_DATE_DETECT] comparison_kw={has_comparison_keyword}, multi_dates={has_multiple_dates}, result={result}")
    
    return result


def _detect_chart_type(user_query: str) -> str:
    """
    Detect what type of chart the user wants.
    
    Returns:
        str: 'bar', 'stacked_bar', 'stacked_line', or 'line' (default)
    """
    query_lower = user_query.lower()
    
    # Check for stacked charts first (more specific)
    if 'stacked bar' in query_lower or 'stack bar' in query_lower:
        return 'stacked_bar'
    elif 'stacked line' in query_lower or 'stack line' in query_lower or 'stacked' in query_lower:
        return 'stacked_line'
    
    # Check for bar chart
    if 'bar chart' in query_lower or 'bar graph' in query_lower or 'barchart' in query_lower or 'bar' in query_lower:
        return 'bar'
    
    # Default to line chart
    return 'line'


def _detect_long_trend_query(user_prompt: str, data: list = None):
    """
    🔥 CRITICAL FIX: Decide whether query is intraday (block-wise) or interday (daily trend).
    
    ABSOLUTE RULES:
    1. Single date → NEVER aggregate (even if data has 96 rows)
    2. 2-7 dates → NO aggregation (comparison mode)
    3. 8+ dates → Aggregate to daily averages
    4. Block-level data (96 rows/day) → NEVER aggregate
    """
    
    # Extract dates from prompt
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", user_prompt)
    
    # 🔒 HARD RULE 1: single date → NEVER aggregate (even if data has multiple rows)
    if len(dates) == 1:
        print(f"[TREND_DETECT] ✅ Single date found: {dates[0]} → NO aggregation (RULE 1)")
        return {
            "is_long_trend": False,
            "aggregation_needed": False,
            "date_range_days": 1,
            "reason": "single_date"
        }
    
    # 🔒 HARD RULE 2: If data provided, check actual structure
    if data and len(data) > 0:
        # Check if data is already aggregated (one row per date)
        if 'date' in data[0]:
            unique_dates = set(str(row['date']) for row in data)
            rows_per_date = len(data) / len(unique_dates) if unique_dates else 0
            
            print(f"[TREND_DETECT] 📊 Data check: {len(data)} rows, {len(unique_dates)} dates, {rows_per_date:.1f} rows/date")
            
            # If we have ~96 rows per date, it's already block-level → don't aggregate
            if rows_per_date > 80:
                print(f"[TREND_DETECT] ✅ Block-level data detected ({rows_per_date:.0f} rows/date) → NO aggregation (RULE 2)")
                return {
                    "is_long_trend": False,
                    "aggregation_needed": False,
                    "date_range_days": len(unique_dates),
                    "reason": "block_level_data"
                }
            
            # If single date with block data
            if len(unique_dates) == 1 and len(data) > 1:
                print(f"[TREND_DETECT] ✅ Single date with {len(data)} blocks → NO aggregation (RULE 2b)")
                return {
                    "is_long_trend": False,
                    "aggregation_needed": False,
                    "date_range_days": 1,
                    "reason": "single_date_block_data"
                }
    
    # Multiple dates in prompt → check range
    if len(dates) >= 2:
        d1 = datetime.strptime(dates[0], "%Y-%m-%d")
        d2 = datetime.strptime(dates[-1], "%Y-%m-%d")
        delta_days = abs((d2 - d1).days) + 1
        
        # 🔒 RULE 3: Only aggregate if range >= 8 days
        # 2-7 days = comparison mode (no aggregation)
        if delta_days < 8:
            print(f"[TREND_DETECT] ✅ Short range ({delta_days} days) → NO aggregation - comparison mode (RULE 3)")
            return {
                "is_long_trend": False,
                "aggregation_needed": False,
                "date_range_days": delta_days,
                "reason": "short_comparison"
            }
        else:
            print(f"[TREND_DETECT] 📈 Long range ({delta_days} days) → AGGREGATION needed (RULE 4)")
            return {
                "is_long_trend": True,
                "aggregation_needed": True,
                "date_range_days": delta_days,
                "reason": "long_range"
            }
    
    # No dates found in prompt → check for keywords
    prompt_lower = user_prompt.lower()
    
    # Monthly/weekly aggregation keywords
    if any(kw in prompt_lower for kw in ['monthly', 'by month', 'per month', 'month-wise']):
        print(f"[TREND_DETECT] 📅 Monthly aggregation keyword found → AGGREGATION needed")
        return {
            "is_long_trend": True,
            "aggregation_needed": True,
            "date_range_days": 30,
            "reason": "monthly_keyword"
        }
    
    if any(kw in prompt_lower for kw in ['weekly', 'by week', 'per week']):
        print(f"[TREND_DETECT] 📅 Weekly aggregation keyword found → AGGREGATION needed")
        return {
            "is_long_trend": True,
            "aggregation_needed": True,
            "date_range_days": 7,
            "reason": "weekly_keyword"
        }
    
    # Default: no aggregation
    print(f"[TREND_DETECT] ✅ No specific detection → NO aggregation (default)")
    return {
        "is_long_trend": False,
        "aggregation_needed": False,
        "date_range_days": 0,
        "reason": "default"
    }


# ------------------------------------------------------
# IMPROVED: DYNAMIC PLOT CODE GENERATION
# ------------------------------------------------------

def generate_plot_code_improved(sql: str, data: list, user_query: str) -> str:
    """
    IMPROVED: Generate HIGH-QUALITY matplotlib code based on data structure.
    Uses smart detection to create the best visualization.
    """
    
    if not data:
        raise ValueError("No data to plot")
    
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Data must be a non-empty list")
    
    if not isinstance(data[0], dict):
        raise ValueError("Data rows must be dictionaries")
    
    columns = list(data[0].keys())
    sample_row = data[0]
    
    print(f"[PLOT_GEN] Analyzing data structure...")
    print(f"[PLOT_GEN] Columns: {columns}")
    print(f"[PLOT_GEN] Sample: {sample_row}")
    print(f"[PLOT_GEN] Row count: {len(data)}")
    print(f"[PLOT_GEN] User query: {user_query}")
    
    # Detect chart type from user query
    chart_type = _detect_chart_type(user_query)
    print(f"[PLOT_GEN] Detected chart type: {chart_type}")
    
    # Detect data type
    has_datetime = any('datetime' in col.lower() for col in columns)
    has_date = any(col.lower() == 'date' for col in columns)
    has_actual = any('actual' in col.lower() or col == 'demand' for col in columns)
    has_forecast = any('forecast' in col.lower() for col in columns)
    has_block = 'block' in columns
    has_avg_demand = 'avg_demand' in columns
    
    # Check for long-term trend vs short comparison
    trend_info = _detect_long_trend_query(user_query, data)
    is_multi_date_comparison = _detect_multi_date_comparison(data, user_query)
    
    # Check for actual vs forecast comparison
    is_actual_vs_forecast = has_actual and has_forecast
    
    is_timeseries = has_datetime or has_date
    
    print(f"[PLOT_GEN] Detection: timeseries={is_timeseries}, actual_vs_forecast={is_actual_vs_forecast}, multi_date={is_multi_date_comparison}, long_trend={trend_info['is_long_trend']}")
    print(f"[PLOT_GEN] Aggregation needed: {trend_info['aggregation_needed']}")
    
    # Generate appropriate plot code
    if trend_info['aggregation_needed'] or has_avg_demand:
        # Long-term aggregated trend
        print("[PLOT_GEN] → Using aggregated trend plot")
        return _generate_aggregated_trend_plot(columns, data, user_query, trend_info, chart_type)
    elif is_multi_date_comparison:
        print("[PLOT_GEN] → Using multi-date comparison plot")
        return _generate_multi_date_comparison_plot(columns, data, user_query, chart_type)
    elif is_actual_vs_forecast:
        print("[PLOT_GEN] → Using actual vs forecast comparison plot")
        return _generate_comparison_plot(columns, len(data), chart_type)
    elif is_timeseries:
        print("[PLOT_GEN] → Using timeseries plot")
        return _generate_timeseries_plot(columns, len(data), has_datetime, chart_type)
    elif has_block:
        print("[PLOT_GEN] → Using block plot")
        return _generate_block_plot(columns, len(data), chart_type)
    else:
        print("[PLOT_GEN] → Using generic plot")
        return _generate_generic_plot(columns, len(data), chart_type)


def _generate_multi_date_comparison_plot(columns: list, data: list, user_query: str, chart_type: str = 'line') -> str:
    """
    IMPROVED: Generate side-by-side comparison for multiple dates.
    FULLY DYNAMIC - works with any value column.
    """
    
    # Dynamically find the value column
    exclude_cols = ['datetime', 'date', 'block', 'name', 'entrydatetime', 'model_id', 'generated_at', 'prediction_date']
    value_columns = [col for col in columns if col not in exclude_cols]
    
    if not value_columns:
        value_columns = [col for col in columns if 'demand' in col.lower()]
    
    if not value_columns:
        raise ValueError(f"Cannot find value column in: {columns}")
    
    value_col = value_columns[0]
    use_bars = chart_type == 'bar'
    
    print(f"[PLOT_GEN] Multi-date plot - Value column: {value_col}, chart_type: {chart_type}")
    
    return f"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import defaultdict

fig, ax = plt.subplots(figsize=(16, 8))

# Group data by date
data_by_date = defaultdict(lambda: {{'x': [], 'y': []}})

for row in data:
    if not isinstance(row, dict):
        continue
    
    date_str = str(row.get('date', ''))
    if not date_str:
        continue
    
    # Try to extract time index (block or datetime)
    if 'block' in row and row['block'] is not None:
        try:
            x_value = int(row['block'])
        except (ValueError, TypeError):
            x_value = len(data_by_date[date_str]['x'])
    elif 'datetime' in row and row['datetime'] is not None:
        try:
            dt = datetime.fromisoformat(str(row['datetime']).replace('Z', '+00:00'))
            x_value = dt.hour * 4 + dt.minute // 15  # Convert to block (0-95)
        except:
            x_value = len(data_by_date[date_str]['x'])
    else:
        x_value = len(data_by_date[date_str]['x'])
    
    # Get value dynamically
    y_value = None
    try:
        y_value = float(row['{value_col}'])
    except (KeyError, ValueError, TypeError):
        for col_name in ['demand', 'actual_demand', 'forecasted_demand', 'predicted_demand', 'value']:
            if col_name in row and row[col_name] is not None:
                try:
                    y_value = float(row[col_name])
                    break
                except (ValueError, TypeError):
                    continue
    
    if y_value is not None:
        data_by_date[date_str]['x'].append(x_value)
        data_by_date[date_str]['y'].append(y_value)

# Sort dates for consistent ordering
sorted_dates = sorted(list(data_by_date.keys()))

if not sorted_dates:
    raise ValueError("No valid data to plot")

# Define colors for different dates
colors = ['#2E86AB', '#F18F01', '#06A77D', '#A23B72', '#D00000', '#7209B7']

# Plot each date
for idx, date_str in enumerate(sorted_dates):
    color = colors[idx % len(colors)]
    x_vals = data_by_date[date_str]['x']
    y_vals = data_by_date[date_str]['y']
    
    if not x_vals or not y_vals:
        continue
    
    if {use_bars}:
        width = 0.8 / len(sorted_dates) if len(sorted_dates) > 0 else 0.8
        offset = (idx - len(sorted_dates)/2 + 0.5) * width
        positions = [x + offset for x in x_vals]
        
        ax.bar(positions, y_vals,
               width=width,
               color=color,
               alpha=0.8,
               edgecolor='black',
               linewidth=1,
               label=f'Date: {{date_str}}')
    else:
        ax.plot(x_vals, y_vals,
                linewidth=2.5,
                color=color,
                marker='o',
                markersize=5,
                markerfacecolor='white',
                markeredgecolor=color,
                markeredgewidth=2,
                label=f'Date: {{date_str}}',
                alpha=0.9)

# Styling
value_label = '{value_col}'.replace('_', ' ').title()
ax.set_title(f'{{value_label}} Comparison Across Dates', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Time Block / Hour', fontsize=13, fontweight='bold')
ax.set_ylabel(f'{{value_label}} (MW)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1, axis='y')
ax.legend(loc='best', fontsize=11, framealpha=0.9, shadow=True)

plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
plt.close()
"""


def _generate_aggregated_trend_plot(columns: list, data: list, user_query: str, trend_info: dict, chart_type: str = 'line') -> str:
    """
    NEW FUNCTION: Generate daily aggregated trend plot for long-term data (8+ days).
    Shows daily averages instead of all 96 blocks per day.
    
    🔥 CRITICAL: This function should ONLY be called for aggregated data!
    """
    
    # CRITICAL VALIDATION: Check if data is actually aggregated (one row per date)
    if 'date' in data[0]:
        unique_dates = set(str(row['date']) for row in data)
        rows_per_date = len(data) / len(unique_dates) if unique_dates else 0
        
        print(f"[PLOT_GEN] Data validation:")
        print(f"  - Total rows: {len(data)}")
        print(f"  - Unique dates: {len(unique_dates)}")
        print(f"  - Rows per date: {rows_per_date:.1f}")
        
        if len(unique_dates) != len(data):
            # Data is NOT aggregated - multiple rows per date
            print(f"[PLOT_GEN] ⚠️ WARNING: Data has {len(data)} rows but only {len(unique_dates)} unique dates")
            print(f"[PLOT_GEN] ⚠️ Performing Python-side aggregation...")
            
            # Aggregate data ourselves
            from collections import defaultdict
            aggregated = defaultdict(lambda: {'sum': 0, 'count': 0, 'min': float('inf'), 'max': float('-inf')})
            
            for row in data:
                date_key = str(row['date'])
                # Try multiple column names
                value = None
                for col in ['demand', 'avg_demand', 'forecasted_demand', 'value']:
                    if col in row and row[col] is not None:
                        value = float(row[col])
                        break
                
                if value is None:
                    print(f"[PLOT_GEN] ⚠️ Could not find value in row: {row}")
                    continue
                
                aggregated[date_key]['sum'] += value
                aggregated[date_key]['count'] += 1
                aggregated[date_key]['min'] = min(aggregated[date_key]['min'], value)
                aggregated[date_key]['max'] = max(aggregated[date_key]['max'], value)
            
            # Rebuild data with daily averages
            data = [
                {
                    'date': date_key,
                    'avg_demand': agg['sum'] / agg['count'],
                    'min_demand': agg['min'],
                    'max_demand': agg['max']
                }
                for date_key, agg in sorted(aggregated.items())
            ]
            
            print(f"[PLOT_GEN] ✅ Aggregated to {len(data)} daily values")
            columns = list(data[0].keys())
        else:
            print(f"[PLOT_GEN] ✅ Data is already properly aggregated (1 row per date)")
    
    # Find value columns
    exclude_cols = ['date', 'datetime', 'block', 'name', 'entrydatetime', 'model_id', 'month', 'blocks']
    value_columns = [col for col in columns if col not in exclude_cols]
    
    if not value_columns:
        raise ValueError(f"Cannot find value column in: {columns}")
    
    # Check if data is already aggregated (has avg_demand, etc.)
    is_aggregated = any('avg' in col.lower() for col in columns)
    
    # Determine if multi-month comparison
    has_month_col = 'month' in columns
    is_multi_month = False
    
    if has_month_col:
        unique_months = set(row.get('month') for row in data if row.get('month') is not None)
        is_multi_month = len(unique_months) > 1
    
    # Find primary value column
    if 'avg_demand' in columns:
        value_col = 'avg_demand'
    elif 'demand' in columns:
        value_col = 'demand'
    else:
        value_col = value_columns[0]
    
    print(f"[PLOT_GEN] Final plot config:")
    print(f"  - value_col: {value_col}")
    print(f"  - multi_month: {is_multi_month}")
    print(f"  - data points: {len(data)}")
    
    if is_multi_month:
        # Multi-month comparison (e.g., March vs April)
        return f"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import defaultdict

fig, ax = plt.subplots(figsize=(16, 8))

# Group data by month
data_by_month = defaultdict(lambda: {{'dates': [], 'values': []}})

for row in data:
    date_obj = datetime.strptime(str(row['date']), '%Y-%m-%d')
    month_name = date_obj.strftime('%B %Y')
    
    data_by_month[month_name]['dates'].append(date_obj)
    data_by_month[month_name]['values'].append(float(row['{value_col}']))

# Sort months
sorted_months = sorted(list(data_by_month.keys()), key=lambda x: datetime.strptime(x, '%B %Y'))

# Define colors
colors = ['#2E86AB', '#F18F01', '#06A77D', '#A23B72', '#D00000', '#7209B7']

# Plot each month
for idx, month_name in enumerate(sorted_months):
    color = colors[idx % len(colors)]
    dates = data_by_month[month_name]['dates']
    values = data_by_month[month_name]['values']
    
    ax.plot(dates, values,
            linewidth=2.5,
            color=color,
            marker='o',
            markersize=4,
            markerfacecolor='white',
            markeredgecolor=color,
            markeredgewidth=1.5,
            label=month_name,
            alpha=0.9)

# Styling
ax.set_title('Daily Average Demand Trend Comparison', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Daily Demand (MW)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.legend(loc='best', fontsize=11, framealpha=0.9, shadow=True)

# Format x-axis
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
plt.close()
"""
    else:
        # Single period trend (e.g., one month)
        if chart_type == 'bar':
            # BAR CHART version
            return f"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

fig, ax = plt.subplots(figsize=(14, 7))

# Extract dates and values
dates = [datetime.strptime(str(row['date']), '%Y-%m-%d') for row in data]
values = [float(row['{value_col}']) for row in data]

# Bar chart
ax.bar(dates, values,
       width=0.8,
       color='#2E86AB',
       alpha=0.8,
       edgecolor='black',
       linewidth=1,
       label='Daily Average Demand')

# Add min/max error bars if available
if 'min_demand' in data[0] and 'max_demand' in data[0]:
    min_values = [float(row['min_demand']) for row in data]
    max_values = [float(row['max_demand']) for row in data]
    
    errors = [[v - m for v, m in zip(values, min_values)],
              [m - v for m, v in zip(max_values, values)]]
    
    ax.errorbar(dates, values, yerr=errors,
                fmt='none',
                ecolor='gray',
                alpha=0.5,
                capsize=3,
                label='Min-Max Range')

# Styling
ax.set_title('Daily Average Demand Trend (Bar Chart)',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Daily Demand (MW)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1, axis='y')
ax.legend(loc='best', fontsize=11, framealpha=0.9)

# Format x-axis
num_days = len(dates)
if num_days <= 10:
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
elif num_days <= 31:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
else:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
plt.close()
"""
        else:
            # LINE CHART version (default)
            return f"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

fig, ax = plt.subplots(figsize=(14, 7))

# Extract dates and values
dates = [datetime.strptime(str(row['date']), '%Y-%m-%d') for row in data]
values = [float(row['{value_col}']) for row in data]

# Main line plot
ax.plot(dates, values,
        linewidth=2.5,
        color='#2E86AB',
        marker='o',
        markersize=5,
        markerfacecolor='white',
        markeredgecolor='#2E86AB',
        markeredgewidth=2,
        label='Daily Average Demand',
        zorder=3)

# Add min/max envelope if available
if 'min_demand' in data[0] and 'max_demand' in data[0]:
    min_values = [float(row['min_demand']) for row in data]
    max_values = [float(row['max_demand']) for row in data]
    
    ax.fill_between(dates, min_values, max_values,
                    alpha=0.2,
                    color='#2E86AB',
                    label='Min-Max Range',
                    zorder=1)

# Styling
ax.set_title('Daily Average Demand Trend',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.set_ylabel('Average Daily Demand (MW)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1, zorder=2)
ax.legend(loc='best', fontsize=11, framealpha=0.9)

# Format x-axis
num_days = len(dates)
if num_days <= 10:
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
elif num_days <= 31:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
else:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
plt.close()
"""


def _generate_timeseries_plot(columns: list, row_count: int, has_datetime: bool, chart_type: str = 'line') -> str:
    """
    🔥 IMPROVED: Generate time-series plot for SINGLE-DATE with ALL BLOCKS
    Supports both LINE and BAR charts for intraday demand patterns.
    """
    
    exclude_cols = ['datetime', 'date', 'block', 'name', 'entrydatetime', 'model_id', 'generated_at', 'prediction_date']
    value_columns = [col for col in columns if col not in exclude_cols]
    
    if not value_columns:
        value_columns = [col for col in columns if 'demand' in col.lower() or 'value' in col.lower()]
    
    if not value_columns:
        raise ValueError(f"Cannot find value column in: {columns}")
    
    value_col = value_columns[0]
    time_col = 'datetime' if has_datetime else 'date'
    
    print(f"[PLOT_GEN] ⭐ Timeseries plot - Value: {value_col}, Time: {time_col}, Chart: {chart_type}, Rows: {row_count}")
    
    if chart_type == 'bar':
        # BAR CHART version - PERFECT for single-date 96-block data
        return f"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

fig, ax = plt.subplots(figsize=(16, 8))

# Extract data with error handling
x_values = []
y_values = []
use_dates = True

for row in data:
    if not isinstance(row, dict):
        continue
    
    # Extract x value (time)
    time_val = row.get('{time_col}')
    if time_val is None:
        continue
    
    try:
        dt = datetime.fromisoformat(str(time_val).replace('Z', '+00:00'))
        x_values.append(dt)
    except:
        use_dates = False
        x_values.append(len(x_values))
    
    # Extract y value (demand)
    demand_val = row.get('{value_col}')
    if demand_val is None:
        y_values.append(0)
    else:
        try:
            y_values.append(float(demand_val))
        except (ValueError, TypeError):
            y_values.append(0)

if not x_values or not y_values:
    raise ValueError("No valid data to plot")

# Create BAR chart
if use_dates:
    # Calculate appropriate bar width based on number of data points
    if {row_count} > 80:
        width = 0.008  # Very narrow bars for 96 blocks (15-min intervals)
    elif {row_count} > 50:
        width = 0.015
    else:
        width = 0.025
    
    ax.bar(x_values, y_values,
           width=width,
           color='#2E86AB',
           alpha=0.85,
           edgecolor='#1a5875',
           linewidth=0.8,
           label='{value_col.replace("_", " ").title()}')
    
    # Format x-axis based on data density
    if {row_count} > 80:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    elif {row_count} > 50:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.xticks(rotation=45, ha='right')
else:
    ax.bar(x_values, y_values,
           color='#2E86AB',
           alpha=0.85,
           edgecolor='#1a5875',
           linewidth=0.8)

# Styling
value_label = '{value_col.replace("_", " ").title()}'
ax.set_title(f'{{value_label}} Over Time (Bar Chart)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Time', fontsize=13, fontweight='bold')
ax.set_ylabel(f'{{value_label}} (MW)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1, axis='y')
ax.legend(loc='best', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
plt.close()
"""
    else:
        # LINE CHART version (default)
        return f"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

fig, ax = plt.subplots(figsize=(16, 8))

# Extract data with error handling
x_values = []
y_values = []
use_dates = True

for row in data:
    if not isinstance(row, dict):
        continue
    
    # Extract x value (time)
    time_val = row.get('{time_col}')
    if time_val is None:
        continue
    
    try:
        dt = datetime.fromisoformat(str(time_val).replace('Z', '+00:00'))
        x_values.append(dt)
    except:
        use_dates = False
        x_values.append(len(x_values))
    
    # Extract y value (demand)
    demand_val = row.get('{value_col}')
    if demand_val is None:
        y_values.append(0)
    else:
        try:
            y_values.append(float(demand_val))
        except (ValueError, TypeError):
            y_values.append(0)

if not x_values or not y_values:
    raise ValueError("No valid data to plot")

# Create plot - LINE with POINTS
if use_dates:
    ax.plot(x_values, y_values, 
            linewidth=2.5, 
            color='#2E86AB', 
            marker='o', 
            markersize=4,
            markerfacecolor='white',
            markeredgecolor='#2E86AB',
            markeredgewidth=1.5,
            label='{value_col.replace("_", " ").title()}')
    
    if {row_count} > 50:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.xticks(rotation=45, ha='right')
else:
    ax.plot(x_values, y_values,
            linewidth=2.5,
            color='#2E86AB',
            marker='o',
            markersize=4,
            markerfacecolor='white',
            markeredgecolor='#2E86AB',
            markeredgewidth=1.5)

# Styling
value_label = '{value_col.replace("_", " ").title()}'
ax.set_title(f'{{value_label}} Over Time', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Time', fontsize=13, fontweight='bold')
ax.set_ylabel(f'{{value_label}} (MW)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)
ax.legend(loc='best', fontsize=11, framealpha=0.9)

# Add value labels for small datasets
if {row_count} <= 20:
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        if i % 2 == 0:
            ax.annotate(f'{{y:.1f}}', 
                       xy=(x, y), 
                       xytext=(0, 10),
                       textcoords='offset points',
                       ha='center',
                       fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', 
                                edgecolor='gray',
                                alpha=0.8))

plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
plt.close()
"""


def _generate_comparison_plot(columns: list, row_count: int, chart_type: str = 'line') -> str:
    """IMPROVED: Generate comparison plot - FULLY DYNAMIC"""
    
    exclude_cols = ['datetime', 'date', 'block', 'name', 'entrydatetime', 'model_id', 'generated_at', 'prediction_date']
    value_columns = [col for col in columns if col not in exclude_cols]
    
    if len(value_columns) < 2:
        raise ValueError(f"Need at least 2 value columns for comparison, found: {value_columns}")
    
    col1 = value_columns[0]
    col2 = value_columns[1]
    
    time_col = 'datetime' if 'datetime' in columns else 'date'
    
    # Determine colors based on column names
    if 'actual' in col1.lower():
        color1, label1 = '#2E86AB', col1.replace('_', ' ').title()
    elif 'forecast' in col1.lower() or 'predict' in col1.lower():
        color1, label1 = '#F18F01', col1.replace('_', ' ').title()
    else:
        color1, label1 = '#2E86AB', col1.replace('_', ' ').title()
    
    if 'forecast' in col2.lower() or 'predict' in col2.lower():
        color2, label2 = '#F18F01', col2.replace('_', ' ').title()
    elif 'actual' in col2.lower():
        color2, label2 = '#2E86AB', col2.replace('_', ' ').title()
    else:
        color2, label2 = '#06A77D', col2.replace('_', ' ').title()
    
    print(f"[PLOT_GEN] Comparison plot - Col1: {col1}, Col2: {col2}, Chart: {chart_type}")
    
    return f"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

fig, ax = plt.subplots(figsize=(14, 7))

# Extract data with error handling
x_values = []
values1 = []
values2 = []
use_dates = True

for row in data:
    if not isinstance(row, dict):
        continue
    
    # Extract x value
    time_val = row.get('{time_col}')
    if time_val is None:
        continue
    
    try:
        dt = datetime.fromisoformat(str(time_val).replace('Z', '+00:00'))
        x_values.append(dt)
    except:
        use_dates = False
        x_values.append(len(x_values))
    
    # Extract y values
    val1 = row.get('{col1}')
    val2 = row.get('{col2}')
    
    try:
        values1.append(float(val1) if val1 is not None else 0)
    except (ValueError, TypeError):
        values1.append(0)
    
    try:
        values2.append(float(val2) if val2 is not None else 0)
    except (ValueError, TypeError):
        values2.append(0)

if not x_values or not values1 or not values2:
    raise ValueError("No valid data to plot")

# Plot both lines with points
ax.plot(x_values, values1,
        linewidth=2.5,
        color='{color1}',
        marker='o',
        markersize=5,
        markerfacecolor='white',
        markeredgecolor='{color1}',
        markeredgewidth=2,
        label='{label1}',
        zorder=3)

ax.plot(x_values, values2,
        linewidth=2.5,
        color='{color2}',
        marker='s',
        markersize=5,
        markerfacecolor='white',
        markeredgecolor='{color2}',
        markeredgewidth=2,
        label='{label2}',
        zorder=3)

# Styling
if use_dates:
    if {row_count} > 50:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xticks(rotation=45, ha='right')

ax.set_title('{label1} vs {label2}', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Time', fontsize=13, fontweight='bold')
ax.set_ylabel('Value (MW)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1, zorder=1)
ax.legend(loc='best', fontsize=12, framealpha=0.9, shadow=True)

# Fill between for visual difference
ax.fill_between(x_values, values1, values2, 
                alpha=0.2, color='gray', zorder=2)

plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
plt.close()
"""


def _generate_block_plot(columns: list, row_count: int, chart_type: str = 'bar') -> str:
    """Generate block-wise bar chart"""
    
    exclude_cols = ['block', 'date', 'datetime', 'name', 'entrydatetime', 'model_id', 'generated_at', 'prediction_date']
    value_columns = [col for col in columns if col not in exclude_cols]
    
    if not value_columns:
        raise ValueError(f"Cannot find value column in: {columns}")
    
    value_col = value_columns[0]
    
    return f"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 7))

# Extract data with error handling
blocks = []
values = []

for row in data:
    if not isinstance(row, dict):
        continue
    
    block_val = row.get('block')
    demand_val = row.get('{value_col}')
    
    if block_val is None or demand_val is None:
        continue
    
    try:
        blocks.append(int(block_val))
        values.append(float(demand_val))
    except (ValueError, TypeError):
        continue

if not blocks or not values:
    raise ValueError("No valid data to plot")

# Bar chart
bars = ax.bar(blocks, values, 
              color='#06A77D', 
              alpha=0.8, 
              edgecolor='black',
              linewidth=1.5)

# Color gradient
if values:
    max_val = max(values)
    if max_val > 0:
        colors = plt.cm.viridis([v/max_val for v in values])
        for bar, color in zip(bars, colors):
            bar.set_color(color)

ax.set_title('{value_col.replace("_", " ").title()} by Block', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Block Number', fontsize=13, fontweight='bold')
ax.set_ylabel('{value_col.replace("_", " ").title()}', 
              fontsize=13, fontweight='bold')
ax.grid(True, axis='y', alpha=0.3, linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
plt.close()
"""


def _generate_generic_plot(columns: list, row_count: int, chart_type: str = 'line') -> str:
    """Fallback generic plot"""
    
    if len(columns) == 0:
        raise ValueError("No columns in data")
    
    x_col = columns[0]
    y_col = columns[1] if len(columns) > 1 else columns[0]
    
    return f"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 7))

# Extract data
y_values = []
for row in data:
    if not isinstance(row, dict):
        continue
    
    val = row.get('{y_col}')
    try:
        y_values.append(float(val) if val is not None else 0)
    except (ValueError, TypeError):
        y_values.append(0)

if not y_values:
    raise ValueError("No valid data to plot")

x_values = list(range(len(y_values)))

ax.plot(x_values, y_values,
        linewidth=2.5,
        color='#2E86AB',
        marker='o',
        markersize=5,
        markerfacecolor='white',
        markeredgecolor='#2E86AB',
        markeredgewidth=2)

ax.set_title('Data Visualization', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Index', fontsize=13, fontweight='bold')
ax.set_ylabel('{y_col.replace("_", " ").title()}', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
plt.close()
"""


# ------------------------------------------------------
# MAIN NL → SQL TOOL - IMPROVED
# ------------------------------------------------------

@tool
def nl_to_sql_db_tool(user_request: str) -> dict:
    """
    IMPROVED: Convert natural language to SQL and execute it.
    Returns data ready for visualization.
    """
    try:
        print(f"[NL2SQL] Processing: {user_request[:100]}...")
        
        # Generate SQL
        sql = generate_sql(user_request)
        print(f"[NL2SQL] Raw SQL: {sql[:200]}...")
        
        # Clean and validate
        sql = strip_markdown(sql)
        sql = validate_sql(sql)
        sql = normalize_invalid_dates(sql)
        sql = replace_relative_dates_with_logical_now(sql)
        sql = clamp_dates_to_availability(sql)
        
        print(f"[NL2SQL] Final SQL: {sql}")
        
        # Execute
        result = execute_query(sql)

        if not result["ok"]:
            print(f"[NL2SQL] Query failed: {result.get('error')}")
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

        print(f"[NL2SQL] Success: {result['row_count']} rows")
        return {
            "ok": True,
            "sql": sql,
            "rows": result["rows"],
            "row_count": result["row_count"],
            "sample_rows": result["sample_rows"],
        }

    except Exception as e:
        print(f"[NL2SQL] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"ok": False, "error": str(e)}


# ------------------------------------------------------
# GRAPH PLOTTING TOOL - DRAMATICALLY IMPROVED
# ------------------------------------------------------
@tool  
def graph_plotting_tool(sql: str, user_query: str = "") -> dict:
    """
    IMPROVED: Execute SQL, fetch data, and generate high-quality visualization.
    ALWAYS queries the database fresh.
    """
    try:
        print(f"[GRAPH] Starting visualization pipeline...")
        print(f"[GRAPH] SQL: {sql[:150]}...")
        print(f"[GRAPH] User query: {user_query[:100]}...")
        
        # Validate inputs
        if not sql or not isinstance(sql, str):
            return {
                "ok": False,
                "error": "Invalid SQL query provided",
            }
        
        # Execute query
        sql = adapt_sql_for_db(sql)
        result = execute_query(sql)
        
        if not result["ok"]:
            print(f"[GRAPH] Query failed: {result.get('error')}")
            return {
                "ok": False,
                "error": f"Database query failed: {result.get('error')}",
                "sql": sql,
            }

        data = result["rows"]
        if not data or len(data) == 0:
            print(f"[GRAPH] No data returned")
            return {
                "ok": False,
                "error": "No data to plot",
                "sql": sql,
            }

        print(f"[GRAPH] Retrieved {len(data)} rows")
        
        # Validate data structure
        if not isinstance(data, list) or not isinstance(data[0], dict):
            return {
                "ok": False,
                "error": "Invalid data structure returned from database",
                "sql": sql,
            }
        
        # IMPROVED: Generate plot code with user_query context
        plot_code = generate_plot_code_improved(sql, data, user_query)
        print(f"[GRAPH] Generated plot code ({len(plot_code)} chars)")
        
        # Execute plot code
        buffer = io.BytesIO()
        
        exec_globals = {
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
                "zip": zip,
                "min": min,
                "max": max,
                "sorted": sorted,
                "set": set,
                "isinstance": isinstance,
            },
            "plt": plt,
            "mdates": mdates,
            "datetime": datetime,
            "defaultdict": defaultdict,
            "ValueError": ValueError,
        }
        
        exec_locals = {
            "data": data,
            "buffer": buffer,
        }
        
        print(f"[GRAPH] Executing matplotlib code...")
        exec(plot_code, exec_globals, exec_locals)
        print(f"[GRAPH] Plot generated successfully")
        
        # Save plot
        buffer.seek(0)
        image_data = buffer.read()
        
        if len(image_data) == 0:
            return {
                "ok": False,
                "error": "Plot generation produced empty image",
                "sql": sql,
            }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data_visualization_{timestamp}.png"
        filepath = PLOTS_DIR / filename
        
        with open(filepath, "wb") as f:
            f.write(image_data)
        
        print(f"[GRAPH] Saved to: {filepath}")
        
        # Encode for response
        image_base64 = base64.b64encode(image_data).decode()
        
        # Detect plot type
        plot_type = "line"
        if "comparison" in user_query.lower() or "vs" in user_query.lower():
            plot_type = "comparison"
        elif "bar" in plot_code.lower():
            plot_type = "bar"
        
        return {
            "ok": True,
            "image_base64": image_base64,
            "plot_file": str(filepath),
            "generated_plot_code": plot_code,
            "plot_type": plot_type,
            "rows_plotted": len(data),
            "sql_executed": sql,
            "data_points": len(data),
        }

    except Exception as e:
        print(f"[GRAPH] Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "ok": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# ------------------------------------------------------
# AGGREGATION QUERY HELPER
# ------------------------------------------------------

def execute_aggregation_query(original_sql: str) -> dict:
    """
    Executes aggregation SQL generated by data_observation_agent.
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
    """Clean up old plot files"""
    cutoff = time.time() - days_to_keep * 86400
    count = 0
    for file in PLOTS_DIR.glob("*.png"):
        if file.stat().st_mtime < cutoff:
            file.unlink(missing_ok=True)
            count += 1
    if count > 0:
        print(f"[CLEANUP] Removed {count} old plot files")


# ------------------------------------------------------
# PLACEHOLDER TOOLS
# ------------------------------------------------------

@tool
def model_run_tool(run_date: str, model_id: int | None = None, model_path: str | None = None) -> dict:
    """Run the forecasting pipeline for `run_date` and persist results to the DB."""
    try:
        from LF_model_run import run_and_store_forecast

        result = run_and_store_forecast(run_date, model_id=model_id, model_path=model_path)

        if not result.get("ok"):
            return {"ok": False, "error": result.get("error", "unknown")}

        return {
            "ok": True,
            "rows_written": result.get("rows_written", 0),
            "prediction_date": result.get("prediction_date"),
            "model_id": result.get("model_id"),
            "metrics": result.get("metrics"),
        }

    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@tool
def live_api_call_tool(
    api_type: str, endpoint: str = None, params: Dict[str, Any] = None
) -> dict:
    """Call an external live API and return the results"""
    return {
        "ok": True,
        "api_type": api_type,
        "status": "not_implemented",
    }
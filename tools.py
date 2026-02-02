# tools.py - MERGED VERSION with improved graph functionality
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
    IMPROVED: Generate database-specific SQL with better handling for graph queries.
    """
    llm = get_agent_llm("nl_to_sql")

    schema = "lf." if DB_TYPE == "postgresql" else ""
    
    # Detect if this is a comparison/graph query
    is_comparison = any(word in user_prompt.lower() for word in [
        'compare', 'comparison', 'vs', 'versus', 'against', 
        'actual and forecast', 'forecast and actual'
    ])

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
You are a {DB_TYPE.upper()} SQL expert specializing in time-series data queries.

CRITICAL RULES:
- Generate ONLY SELECT queries
- NO DELETE / UPDATE / INSERT / DROP
- Output ONLY SQL - no explanations, no markdown, no backticks
- End query with semicolon
- DO NOT use GROUP BY unless explicitly asked for aggregation/statistics
- DO NOT use SUM/AVG/COUNT unless explicitly asked for aggregation
- For visualization queries, return RAW data: datetime, date, block, demand

AVAILABLE DATA RANGES:
- Actual demand: {ACTUAL_DATA_START.date()} → {ACTUAL_DATA_END.date()}
- Forecasted demand: {FORECAST_DATA_START.date()} → {FORECAST_DATA_END.date()}
- Holidays: {HOLIDAY_START} → {HOLIDAY_END}
- Metrics: {METRICS_START} → {METRICS_END}

TABLES & COLUMNS:
1. {schema}t_actual_demand
   - datetime (timestamp with time data)
   - date (date only)
   - block (integer 1-96, time slot)
   - demand (real number, MW)
   
2. {schema}t_forecasted_demand
   - datetime (timestamp)
   - date (date only)  
   - block (integer 1-96)
   - forecasted_demand (real number, MW)
   - model_id (text)

3. {schema}t_holidays
   - date, name, normal_holiday, special_day

4. {schema}t_metrics
   - date, mape, rmse, model_id

QUERY PATTERNS:

**For Single Date Queries (Trend/Graph):**
```sql
SELECT datetime, date, block, demand 
FROM {schema}t_actual_demand 
WHERE date = '2025-05-15'
ORDER BY datetime;
```

**For Multi-Date Comparison (e.g., "compare 15 august vs 16 august"):**
IMPORTANT: NO GROUP BY, NO SUM - just raw data!
```sql
SELECT datetime, date, block, demand
FROM {schema}t_actual_demand
WHERE date IN ('2025-08-15', '2025-08-16')
ORDER BY date, datetime;
```

**For Actual vs Forecast Comparison (Single Date):**
```sql
SELECT 
    a.datetime,
    a.date,
    a.block,
    a.demand as actual_demand,
    f.forecasted_demand
FROM {schema}t_actual_demand a
JOIN {schema}t_forecasted_demand f 
    ON a.datetime = f.datetime
WHERE a.date = '2025-05-15'
ORDER BY a.datetime;
```

**For Date Range Queries:**
```sql
SELECT datetime, date, block, demand
FROM {schema}t_actual_demand
WHERE date >= '2025-05-01' AND date <= '2025-05-31'
ORDER BY datetime;
```

**For Statistics/Aggregation (ONLY when explicitly asked):**
```sql
SELECT date, AVG(demand) as avg_demand, MAX(demand) as max_demand
FROM {schema}t_actual_demand
WHERE date IN ('2025-08-15', '2025-08-16')
GROUP BY date;
```

IMPORTANT RULES:
1. For multi-date comparisons, use WHERE date IN (...) NOT JOIN
2. Always include datetime, date, block, demand for visualization
3. For comparisons, JOIN on datetime (not date)
4. Always ORDER BY datetime (or date, datetime for multi-date)
5. Use simple column names (demand, not aliases unless comparing actual vs forecast)
6. NEVER use UNION for date comparisons - use WHERE IN instead
7. DO NOT use GROUP BY or aggregation functions unless user explicitly asks for "average", "sum", "total", "statistics"
8. For graphs/charts/visualization: return RAW individual data points, NOT aggregated data

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
# IMPROVED: DYNAMIC PLOT CODE GENERATION
# ------------------------------------------------------

def generate_plot_code_improved(sql: str, data: list, user_query: str) -> str:
    """
    IMPROVED: Generate HIGH-QUALITY matplotlib code based on data structure.
    Uses smart detection to create the best visualization.
    """
    
    if not data:
        raise ValueError("No data to plot")
    
    columns = list(data[0].keys())
    sample_row = data[0]
    
    print(f"[PLOT_GEN] Analyzing data structure...")
    print(f"[PLOT_GEN] Columns: {columns}")
    print(f"[PLOT_GEN] Sample: {sample_row}")
    print(f"[PLOT_GEN] Row count: {len(data)}")
    print(f"[PLOT_GEN] User query: {user_query}")
    
    # Detect data type
    has_datetime = any('datetime' in col.lower() for col in columns)
    has_date = any(col.lower() == 'date' for col in columns)
    has_actual = any('actual' in col.lower() or col == 'demand' for col in columns)
    has_forecast = any('forecast' in col.lower() for col in columns)
    has_block = 'block' in columns
    
    # Check for multi-date comparison (e.g., "compare 15 august vs 16 august")
    is_multi_date_comparison = _detect_multi_date_comparison(data, user_query)
    
    # Check for actual vs forecast comparison
    is_actual_vs_forecast = has_actual and has_forecast
    
    is_timeseries = has_datetime or has_date
    
    print(f"[PLOT_GEN] Detection: timeseries={is_timeseries}, actual_vs_forecast={is_actual_vs_forecast}, multi_date={is_multi_date_comparison}")
    
    # Generate appropriate plot code
    if is_multi_date_comparison:
        return _generate_multi_date_comparison_plot(columns, data, user_query)
    elif is_actual_vs_forecast:
        return _generate_comparison_plot(columns, len(data))
    elif is_timeseries:
        return _generate_timeseries_plot(columns, len(data), has_datetime)
    elif has_block:
        return _generate_block_plot(columns, len(data))
    else:
        return _generate_generic_plot(columns, len(data))


def _detect_multi_date_comparison(data: list, user_query: str) -> bool:
    """
    Detect if this is a multi-date comparison query.
    E.g., "compare 15 august vs 16 august", "15 and 16 august demand"
    """
    comparison_keywords = ['compare', 'vs', 'versus', 'between', 'and']
    query_lower = user_query.lower()
    
    has_comparison_keyword = any(kw in query_lower for kw in comparison_keywords)
    
    if 'date' in data[0]:
        unique_dates = set(str(row['date']) for row in data)
        has_multiple_dates = len(unique_dates) > 1
    else:
        has_multiple_dates = False
    
    columns = list(data[0].keys())
    has_date_col = 'date' in columns
    has_demand_col = any(col in columns for col in ['demand', 'actual_demand', 'forecasted_demand'])
    
    result = has_comparison_keyword and has_multiple_dates and has_date_col and has_demand_col
    
    print(f"[MULTI_DATE_DETECT] comparison_kw={has_comparison_keyword}, multi_dates={has_multiple_dates}, result={result}")
    
    return result


def _generate_multi_date_comparison_plot(columns: list, data: list, user_query: str) -> str:
    """
    IMPROVED: Generate side-by-side comparison for multiple dates.
    FULLY DYNAMIC - works with any value column.
    """
    
    # Dynamically find the value column
    exclude_cols = ['datetime', 'date', 'block', 'name', 'entrydatetime', 'model_id']
    value_columns = [col for col in columns if col not in exclude_cols]
    
    if not value_columns:
        value_columns = [col for col in columns if 'demand' in col.lower()]
    
    if not value_columns:
        raise ValueError(f"Cannot find value column in: {columns}")
    
    value_col = value_columns[0]
    use_bars = 'bar' in user_query.lower()
    
    print(f"[PLOT_GEN] Multi-date plot - Value column: {value_col}, use_bars: {use_bars}")
    
    return f"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import defaultdict

fig, ax = plt.subplots(figsize=(16, 8))

# Group data by date
data_by_date = defaultdict(lambda: {{'x': [], 'y': []}})

for row in data:
    date_str = str(row['date'])
    
    # Try to extract time index (block or datetime)
    if 'block' in row:
        x_value = int(row['block'])
    elif 'datetime' in row:
        try:
            dt = datetime.fromisoformat(str(row['datetime']))
            x_value = dt.hour * 4 + dt.minute // 15  # Convert to block (0-95)
        except:
            x_value = len(data_by_date[date_str]['x'])
    else:
        x_value = len(data_by_date[date_str]['x'])
    
    # Get value dynamically
    try:
        y_value = float(row['{value_col}'])
    except (KeyError, ValueError, TypeError):
        for col_name in ['demand', 'actual_demand', 'forecasted_demand', 'value']:
            if col_name in row:
                y_value = float(row[col_name])
                break
        else:
            y_value = 0
    
    data_by_date[date_str]['x'].append(x_value)
    data_by_date[date_str]['y'].append(y_value)

# Sort dates for consistent ordering
sorted_dates = sorted(list(data_by_date.keys()))

# Define colors for different dates
colors = ['#2E86AB', '#F18F01', '#06A77D', '#A23B72', '#D00000', '#7209B7']

# Plot each date
for idx, date_str in enumerate(sorted_dates):
    color = colors[idx % len(colors)]
    x_vals = data_by_date[date_str]['x']
    y_vals = data_by_date[date_str]['y']
    
    if {use_bars}:
        width = 0.8 / len(sorted_dates)
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


def _generate_timeseries_plot(columns: list, row_count: int, has_datetime: bool) -> str:
    """IMPROVED: Generate time-series line plot - FULLY DYNAMIC"""
    
    exclude_cols = ['datetime', 'date', 'block', 'name', 'entrydatetime', 'model_id']
    value_columns = [col for col in columns if col not in exclude_cols]
    
    if not value_columns:
        value_columns = [col for col in columns if 'demand' in col.lower() or 'value' in col.lower()]
    
    if not value_columns:
        raise ValueError(f"Cannot find value column in: {columns}")
    
    value_col = value_columns[0]
    time_col = 'datetime' if has_datetime else 'date'
    
    print(f"[PLOT_GEN] Timeseries plot - Value: {value_col}, Time: {time_col}")
    
    return f"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

fig, ax = plt.subplots(figsize=(14, 7))

# Extract data
try:
    x_values = [datetime.fromisoformat(str(row['{time_col}'])) for row in data]
    use_dates = True
except:
    x_values = list(range(len(data)))
    use_dates = False

y_values = [float(row['{value_col}']) for row in data]

# Create plot - LINE with POINTS
if use_dates:
    ax.plot(x_values, y_values, 
            linewidth=2.5, 
            color='#2E86AB', 
            marker='o', 
            markersize=5,
            markerfacecolor='white',
            markeredgecolor='#2E86AB',
            markeredgewidth=2,
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
            markersize=5,
            markerfacecolor='white',
            markeredgecolor='#2E86AB',
            markeredgewidth=2)

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


def _generate_comparison_plot(columns: list, row_count: int) -> str:
    """IMPROVED: Generate comparison plot - FULLY DYNAMIC"""
    
    exclude_cols = ['datetime', 'date', 'block', 'name', 'entrydatetime', 'model_id']
    value_columns = [col for col in columns if col not in exclude_cols]
    
    if len(value_columns) < 2:
        raise ValueError(f"Need at least 2 value columns for comparison, found: {value_columns}")
    
    col1 = value_columns[0]
    col2 = value_columns[1]
    
    time_col = 'datetime' if 'datetime' in columns else 'date'
    
    # Determine colors based on column names
    if 'actual' in col1.lower():
        color1, label1 = '#2E86AB', col1.replace('_', ' ').title()
    elif 'forecast' in col1.lower():
        color1, label1 = '#F18F01', col1.replace('_', ' ').title()
    else:
        color1, label1 = '#2E86AB', col1.replace('_', ' ').title()
    
    if 'forecast' in col2.lower():
        color2, label2 = '#F18F01', col2.replace('_', ' ').title()
    elif 'actual' in col2.lower():
        color2, label2 = '#2E86AB', col2.replace('_', ' ').title()
    else:
        color2, label2 = '#06A77D', col2.replace('_', ' ').title()
    
    print(f"[PLOT_GEN] Comparison plot - Col1: {col1}, Col2: {col2}")
    
    return f"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

fig, ax = plt.subplots(figsize=(14, 7))

# Extract data
try:
    x_values = [datetime.fromisoformat(str(row['{time_col}'])) for row in data]
    use_dates = True
except:
    x_values = list(range(len(data)))
    use_dates = False

values1 = [float(row['{col1}']) for row in data]
values2 = [float(row['{col2}']) for row in data]

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


def _generate_block_plot(columns: list, row_count: int) -> str:
    """Generate block-wise bar chart"""
    
    value_col = None
    for col in columns:
        if col not in ['block', 'date', 'datetime']:
            value_col = col
            break
    
    return f"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 7))

blocks = [int(row['block']) for row in data]
values = [float(row['{value_col}']) for row in data]

# Bar chart
bars = ax.bar(blocks, values, 
              color='#06A77D', 
              alpha=0.8, 
              edgecolor='black',
              linewidth=1.5)

# Color gradient
colors = plt.cm.viridis([v/max(values) for v in values])
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


def _generate_generic_plot(columns: list, row_count: int) -> str:
    """Fallback generic plot"""
    
    x_col = columns[0]
    y_col = columns[1] if len(columns) > 1 else columns[0]
    
    return f"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 7))

x_values = list(range(len(data)))
y_values = [float(row['{y_col}']) for row in data]

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
        if not data:
            print(f"[GRAPH] No data returned")
            return {
                "ok": False,
                "error": "No data to plot",
                "sql": sql,
            }

        print(f"[GRAPH] Retrieved {len(data)} rows")
        
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
            },
            "plt": plt,
            "mdates": mdates,
            "datetime": datetime,
            "defaultdict": defaultdict,
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
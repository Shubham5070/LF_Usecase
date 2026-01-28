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
from datetime import datetime
from typing import Dict, Any

import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# Project imports
from db_factory import DatabaseFactory
from agent_llm_config import get_agent_llm

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

    sql = re.sub(r"\blf\.", "", sql)

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
    Generate database-specific SQL using agent-configured LLM.
    """
    llm = get_agent_llm("nl_to_sql")

    if DB_TYPE == "postgresql":
        schema = "lf."
    else:
        schema = ""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""
You are a {DB_TYPE.upper()} SQL expert.
Generate SIMPLE SELECT queries only.

Rules:
- No DELETE / UPDATE / INSERT
- Avoid JOINs unless necessary
- Output ONLY SQL

Tables:
- {schema}t_actual_demand(datetime, date, block, demand, entrydatetime)
- {schema}t_forecasted_demand(datetime, date, block, forecasted_demand, model_id, entrydatetime)
- {schema}t_holidays(date, name, normal_holiday, special_day, entrydatetime)
- {schema}t_metrics(date, mape, rmse, model_id, entrydatetime)
""",
            ),
            ("user", "{user_prompt}"),
        ]
    )

    response = llm.invoke(
        prompt.format_messages(user_prompt=user_prompt)
    ).content.strip()

    return response


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

        result = execute_query(sql, limit=10)
        if not result["ok"]:
            return result

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
def graph_plotting_tool(
    sql: str,
    x_column: str,
    y_column: str,
    plot_type: str = "line",
    title: str = "Data Visualization",
    limit: int = 10000,
) -> dict:
    """
    Plot data from SQL query.
    """
    try:
        sql = adapt_sql_for_db(sql)
        if "limit" not in sql.lower():
            sql = sql.replace(";", f" LIMIT {limit};")

        result = execute_query(sql)
        if not result["ok"] or not result["rows"]:
            raise ValueError("No data returned")

        data = result["rows"]
        x = [r[x_column] for r in data if x_column in r]
        y = [r[y_column] for r in data if y_column in r]

        plt.figure(figsize=(12, 6))
        if plot_type == "line":
            plt.plot(x, y, marker="o")
        elif plot_type == "bar":
            plt.bar(range(len(x)), y)
            plt.xticks(range(len(x)), x)
        elif plot_type == "scatter":
            plt.scatter(x, y)
        else:
            raise ValueError("Invalid plot type")

        plt.title(title)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.xticks(rotation=45)
        plt.tight_layout()

        filename = f"{title.replace(' ', '_')}_{int(time.time())}.png"
        path = PLOTS_DIR / filename
        plt.savefig(path, dpi=150)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=150)
        buffer.seek(0)

        plt.close()

        return {
            "ok": True,
            "filepath": str(path),
            "image_base64": base64.b64encode(buffer.read()).decode(),
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}



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

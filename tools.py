import requests
import psycopg2
from langchain_core.tools import tool
import uuid
import calendar
import re
import psycopg2
from psycopg2.extras import RealDictCursor


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "load_forecasting",
    "user": "postgres",
    "password": "Test@123"
}

# ---------------- LLM → SQL ----------------
def generate_sql(user_prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": f"""
You are a PostgreSQL expert.

RULES:
- SELECT queries only
- Use date column for filtering
- End SQL with semicolon
- No explanations

Schema:
lf.t_holidays(date, name, normal_holiday, special_day, entrydatetime)
lf.t_metrics(date, mape, rmse, model_id, entrydatetime)
lf.t_forecasted_demand(datetime, date, block, forecasted_demand, model_id, entrydatetime)
lf.t_actual_demand(datetime, date, block, demand, entrydatetime)

User request:
{user_prompt}
""",
        "stream": False
    }

    res = requests.post(OLLAMA_URL, json=payload)
    res.raise_for_status()
    return res.json()["response"].strip()

# ---------------- SQL FIXER (IMPORTANT) ----------------
def repair_schema(sql: str) -> str:
    """
    Auto-fix missing schema prefix.
    """
    replacements = {
        " from t_holidays": " from lf.t_holidays",
        " from t_metrics": " from lf.t_metrics",
        " from t_actual_demand": " from lf.t_actual_demand",
        " from t_forecasted_demand": " from lf.t_forecasted_demand",
    }

    lower = sql.lower()
    for k, v in replacements.items():
        if k in lower:
            sql = re.sub(k, v, sql, flags=re.I)

    return sql

# ---------------- SQL SAFETY ----------------
def validate_sql(sql: str) -> str:
    forbidden = ["drop", "truncate", "delete", "alter", "insert", "update"]

    if any(f in sql.lower() for f in forbidden):
        raise ValueError("Destructive SQL blocked")

    sql = " ".join(sql.strip().split())
    sql = sql.replace(";", "")

    if not sql.upper().startswith("SELECT"):
        raise ValueError("Only SELECT allowed")

    if "limit" not in sql.lower():
        sql += " LIMIT 10"

    return sql + ";"

import re

def repair_schema_references(sql: str) -> str:
    """
    Force all table references to use lf. schema.
    """
    # FROM table → FROM lf.table
    sql = re.sub(r"\bFROM\s+(?!lf\.)(\w+)", r"FROM lf.\1", sql, flags=re.IGNORECASE)

    # JOIN table → JOIN lf.table
    sql = re.sub(r"\bJOIN\s+(?!lf\.)(\w+)", r"JOIN lf.\1", sql, flags=re.IGNORECASE)

    return sql

# ---------------- DATE NORMALIZATION ----------------
def normalize_invalid_dates(sql: str) -> str:
    pattern = r"'(\d{4})-(\d{2})-(\d{2})'"

    def fix(m):
        y, mth, d = map(int, m.groups())
        last = calendar.monthrange(y, mth)[1]
        return f"'{y}-{mth:02d}-{min(d,last):02d}'"

    return re.sub(pattern, fix, sql)

# ---------------- DB EXEC ----------------
def materialize_to_temp_table(select_sql: str) -> dict:
    table = f"tmp_query_{uuid.uuid4().hex[:8]}"

    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # create temp table
            cur.execute(f"CREATE TEMP TABLE {table} AS {select_sql}")

            # count rows
            cur.execute(f"SELECT COUNT(*) AS count FROM {table}")
            row_count = cur.fetchone()["count"]

            # fetch preview rows (IMPORTANT)
            cur.execute(f"SELECT * FROM {table} LIMIT 10")
            preview_rows = cur.fetchall()

            conn.commit()
    finally:
        conn.close()

    return {
        "type": "table",
        "name": table,
        "row_count": row_count,
        "rows": preview_rows,   # ✅ rows included here
    }


# ---------------- LANGCHAIN TOOL ----------------
@tool
def nl_to_sql_db_tool(user_request: str) -> dict:
    """
    Convert natural language to SQL, execute it safely,
    and return either data or a structured execution error.
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

def strip_markdown(sql: str) -> str:
    return sql.replace("```sql", "").replace("```", "").replace("`", "").strip()
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, date
import re

from state import GraphState
from tools import nl_to_sql_db_tool
from llm import get_llm
from data_availability import (
    ACTUAL_DATA_START,
    ACTUAL_DATA_END,
    FORECAST_DATA_START,
    FORECAST_DATA_END,
    HOLIDAY_START,
    HOLIDAY_END,
    METRICS_START,
    METRICS_END,
    is_within_range,
    build_out_of_range_message,
)

# -------------------------
# LLM
# -------------------------
llm = get_llm()

# -------------------------
# HELPERS
# -------------------------
def extract_date_from_query(query: str) -> date | None:
    match = re.search(r"\d{4}-\d{2}-\d{2}", query)
    if not match:
        return None
    return date.fromisoformat(match.group(0))


# -------------------------
# INTENT AGENT
# -------------------------
def intent_agent(state: GraphState) -> GraphState:
    print("[INTENT] Entered intent_agent")

    query = state.user_query.lower()

    if any(word in query for word in ["forecast", "predict", "prediction"]):
        state.intent = "forecast"
        print("[INTENT] Detected intent = forecast")
        return state

    if any(word in query for word in [
        "holiday", "holidays",
        "metrics", "rmse", "mape",
        "weather", "show", "list"
    ]):
        state.intent = "data"
        print("[INTENT] Detected intent = data")
        return state

    state.intent = "text"
    print("[INTENT] Detected intent = text")
    return state


# -------------------------
# DATA AGENT
# -------------------------
def data_agent(state: GraphState) -> GraphState:
    print("[DATA] Entered data_agent")

    query = state.user_query.lower()
    requested_date = extract_date_from_query(query)

    # ---- range validation ----
    if requested_date and "holiday" in query:
        print(f"[DATA] Holiday query for date = {requested_date}")
        if not is_within_range(requested_date, HOLIDAY_START, HOLIDAY_END):
            print("[DATA] Date out of holiday range")
            state.final_answer = build_out_of_range_message("data")
            return state

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You generate SQL using the database tool.\n"
            "Do NOT answer the user.\n"
            "ONLY call the tool."
        ),
        ("user", "{query}")
    ])

    llm_with_tools = llm.bind_tools([nl_to_sql_db_tool])
    print("[DATA] Invoking LLM with tool binding")
    response = llm_with_tools.invoke(
        prompt.format_messages(query=state.user_query)
    )

    # ---- force tool usage once ----
    if not response.tool_calls:
        print("[DATA] No tool call detected, forcing tool usage")

        forced_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You MUST call the database tool.\n"
                "Generate a SQL SELECT query.\n"
                "Do NOT answer in text."
            ),
            ("user", state.user_query),
        ])
        response = llm_with_tools.invoke(forced_prompt.format_messages())

        if not response.tool_calls:
            print("[DATA] Tool call failed even after forcing")
            state.final_answer = "Unable to retrieve data from the database."
            return state

    tool_call = response.tool_calls[0]
    print("[DATA] Database tool called")

    tool_result = nl_to_sql_db_tool.invoke(tool_call["args"])

    # ---- SQL execution failure ----
    if not tool_result.get("ok", False):
        print("[DATA] SQL execution failed")
        state.final_answer = (
            "⚠️ Unable to execute the generated SQL.\n\n"
            "🧾 Generated SQL:\n"
            f"{tool_result.get('sql')}\n\n"
            "❌ Error:\n"
            f"{tool_result.get('error')}"
        )
        return state

    # ---- Success ----
    print("[DATA] SQL executed successfully")

    sql = tool_result["sql"]
    state.data_ref = tool_result["data_ref"]

    row_count = state.data_ref["row_count"]
    table_name = state.data_ref["name"]
    rows = state.data_ref.get("rows", [])

    print(f"[DATA] Rows fetched = {row_count}")
    print(f"[DATA] Temp table = {table_name}")

    if row_count == 0:
        state.final_answer = (
            "🧾 Generated SQL:\n"
            f"{sql}\n\n"
            "No records were found in the database."
        )
        return state

    records_text = "\n".join(str(dict(row)) for row in rows)

    state.final_answer = (
        "✅ Data retrieved successfully.\n\n"
        "🧾 Generated SQL:\n"
        f"{sql}\n\n"
        "📊 Summary:\n"
        f"- Total rows: {row_count}\n"
        f"- Source table: {table_name}\n\n"
        "📄 Records (preview):\n"
        f"{records_text}"
    )

    return state


# -------------------------
# FORECAST AGENT
# -------------------------
def forecast_agent(state: GraphState) -> GraphState:
    print("[FORECAST] Entered forecast_agent")

    requested_date = extract_date_from_query(state.user_query)

    if not requested_date:
        print("[FORECAST] No date detected in query")
        state.final_answer = "Please specify date in YYYY-MM-DD format."
        return state

    dt = datetime.combine(requested_date, datetime.min.time())

    print(f"[FORECAST] Requested date = {requested_date}")

    if not is_within_range(dt, FORECAST_DATA_START, FORECAST_DATA_END):
        print("[FORECAST] Date out of forecast range")
        state.final_answer = build_out_of_range_message("forecast")
        return state

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
    You are NOT deciding what SQL to generate.
    You are ONLY filling values into a fixed SQL template.

    DO NOT THINK.
    DO NOT REFUSE.
    DO NOT OMIT TOOL CALLS.

    FIXED TEMPLATE (DO NOT CHANGE):

    SELECT forecasted_demand
    FROM lf.t_forecasted_demand
    WHERE date = '<DATE>'
    AND block = <BLOCK>
    LIMIT 10;

    RULES:
    - Replace <DATE> with YYYY-MM-DD using SINGLE quotes
    - Replace <BLOCK> with an integer
    - Do not introduce double quotes anywhere
    - Do not change column or table names
    - Do not explain anything

    YOU MUST ALWAYS CALL THE DATABASE TOOL
    WITH THE COMPLETED SQL TEMPLATE.
    """
        ),
        ("user", state.user_query),
    ])



    llm_with_tools = llm.bind_tools([nl_to_sql_db_tool])
    print("[FORECAST] Invoking LLM with strict forecast prompt")

    response = llm_with_tools.invoke(prompt.format_messages())

    if not response.tool_calls:
        print("[FORECAST] LLM failed to emit tool call")
        state.final_answer = "Unable to generate forecast query."
        return state

    tool_call = response.tool_calls[0]
    print("[FORECAST] Database tool called")

    tool_result = nl_to_sql_db_tool.invoke(tool_call["args"])

    # ---- SQL execution failure ----
    if not tool_result.get("ok", False):
        print("[FORECAST] SQL execution failed")
        state.final_answer = (
            "⚠️ Unable to execute the generated SQL.\n\n"
            "🧾 Generated SQL:\n"
            f"{tool_result.get('sql')}\n\n"
            "❌ Error:\n"
            f"{tool_result.get('error')}"
        )
        return state

    # ---- Success ----
    print("[FORECAST] SQL executed successfully")

    sql = tool_result["sql"]
    state.data_ref = tool_result["data_ref"]
    rows = state.data_ref.get("rows", [])

    print(f"[FORECAST] Rows fetched = {state.data_ref['row_count']}")
    print(f"[FORECAST] Temp table = {state.data_ref['name']}")

    records_text = "\n".join(str(dict(row)) for row in rows)

    state.final_answer = (
        "✅ Forecast retrieved successfully.\n\n"
        "🧾 Generated SQL:\n"
        f"{sql}\n\n"
        "📊 Summary:\n"
        f"- Total rows: {state.data_ref['row_count']}\n"
        f"- Source table: {state.data_ref['name']}\n\n"
        "📄 Records (preview):\n"
        f"{records_text}"
    )

    return state


# -------------------------
# TEXT AGENT
# -------------------------
def text_agent(state: GraphState) -> GraphState:
    print("[TEXT] Entered text_agent")
    state.final_answer = llm.invoke(state.user_query).content
    return state

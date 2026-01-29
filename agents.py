from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, date
import re
import json
from state import GraphState
from tools import nl_to_sql_db_tool, graph_plotting_tool
from llm import get_llm
from agent_llm_config import get_agent_llm
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


def should_show_technical_details(query: str) -> bool:
    """Determine if user wants to see SQL/technical details"""
    technical_keywords = [
        "sql", "query", "show query", "show sql", "technical",
        "debug", "how did you", "what query", "database", "show me the query"
    ]
    return any(keyword in query.lower() for keyword in technical_keywords)

# -------------------------
# QUERY ANALYSIS AGENT (Entry Point)
# -------------------------
def query_analysis_agent(state: GraphState) -> GraphState:
    """
    Analyzes user query and determines routing.
    Maps to 'Query analysis' node in flowchart.
    """
    print("[QUERY_ANALYSIS] Analyzing user query")
    
    # Get LLM configured for this agent
    llm = get_agent_llm("query_analysis")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a query analysis agent for a load forecasting system.

Your task:
- Classify the user's intent
- Determine required tools and agents

Intent categories:
- "data" - Historical data queries to work or fetch data of actual demand, holidays, metrics
- "forecast" - Future demand predictions from the database provided
- "decision" - Business intelligence, recommendations, insights, for complex action effort queries
- "text" - General questions, explanations, definitions

Tool requirements:
- need_db_call: true if query needs database access (for actual demand, holidays, metrics and forecast value)
- need_graph: true ONLY if user explicitly asks for: trend, plot, graph, chart, visualize, show variation
- need_model_run: true if user wants to train/run a new model
- need_api_call: true if external API data is needed

Rules:
- Forecast queries ALWAYS need DB
- Data queries about actual/historical records need DB like for actual demand, holidays, metrics and forecasting values
- Explanations/definitions do NOT need DB
- "decision" intent may need DB for data-driven insights

Respond ONLY in valid JSON:
{{
  "intent": "<data|forecast|decision|text>",
  "need_db_call": true,
  "need_graph": false,
  "need_model_run": false,
  "need_api_call": false
}}
"""
        ),
        ("user", "{query}")
    ])

    response = llm.invoke(
        prompt.format_messages(query=state.user_query)
    ).content

    print("[QUERY_ANALYSIS] Raw response:", response)

    # Parse JSON response
    try:
        result = json.loads(response)
        state.intent = result.get("intent", "text")
        state.need_db_call = result.get("need_db_call", False)
        state.need_graph = result.get("need_graph", False)
        state.need_model_run = result.get("need_model_run", False)
        state.need_api_call = result.get("need_api_call", False)

    except Exception as e:
        print("[QUERY_ANALYSIS] Parse error:", e)
        state.intent = "text"
        state.need_db_call = False
        state.need_graph = False

    print(
        f"[QUERY_ANALYSIS] intent={state.intent}, "
        f"need_db_call={state.need_db_call}, "
        f"need_graph={state.need_graph}"
    )

    return state

# -------------------------
# INTENT IDENTIFIER (PLANNER)
# -------------------------
def intent_identifier_agent(state: GraphState) -> GraphState:
    """
    Plans execution path based on query analysis.
    Maps to 'Intent Identifier (Planner)' node in flowchart.
    """
    print("[PLANNER] Planning execution path")
    print(f"[PLANNER] Execution plan: intent={state.intent}")
    return state


# -------------------------
# NL2SQL AGENT
# -------------------------
def nl_to_sql_agent(state: GraphState) -> GraphState:
    """
    Converts natural language to SQL and executes query.
    Maps to 'NL to SQL AGENT' node in flowchart.
    """
    print("[NL2SQL] Converting query to SQL")
    print(f"[NL2SQL] Calling DB tool with query: {state.user_query}")
    
    tool_result = nl_to_sql_db_tool.invoke(state.user_query)

    if not tool_result.get("ok", False):
        # Store error for summarization agent to format
        state.data_ref = {
            "ok": False,
            "sql": tool_result.get("sql", ""),
            "error": tool_result.get("error", "Unknown error"),
            "error_type": "sql_execution"
        }
        return state

    state.data_ref = tool_result
    print(f"[NL2SQL] Success: {tool_result.get('row_count', 0)} rows retrieved")
    return state


# -------------------------
# DATA AND OBSERVATION AGENT
# -------------------------
def data_observation_agent(state: GraphState) -> GraphState:
    """
    Handles historical data queries and observations.
    Maps to 'Data and observation AGENT' node in flowchart.
    """
    print("[DATA_OBSERVATION] Processing data query")

    # Check if we already have data from NL2SQL
    if state.data_ref and state.data_ref.get("ok", True):
        sql = state.data_ref.get("sql", "")
        row_count = state.data_ref.get("row_count", 0)
        rows = state.data_ref.get("rows", [])
        sample_rows = state.data_ref.get("sample_rows", [])

        if row_count == 0:
            # Let summarization agent handle this
            state.data_ref["message_type"] = "no_data"
            return state

        print(f"[DATA_OBSERVATION] Dataset has {row_count} rows")

        # For large datasets (>100 rows), generate aggregation query
        if row_count > 100:
            print("[DATA_OBSERVATION] Large dataset detected - generating aggregation")
            
            agg_sql = generate_aggregation_query(sql, state.user_query)
            print(f"[DATA_OBSERVATION] Aggregation SQL: {agg_sql}")
            
            from tools import execute_aggregation_query
            agg_result = execute_aggregation_query(agg_sql)
            
            if agg_result.get("ok"):
                agg_rows = agg_result.get("rows", [])
                state.data_ref["aggregation"] = agg_rows
                state.data_ref["message_type"] = "aggregated_data"
            else:
                state.data_ref["message_type"] = "sample_data"
        else:
            state.data_ref["message_type"] = "small_dataset"

        # Prepare graph data if needed
        if state.need_graph:
            print("[DATA_OBSERVATION] Preparing graph metadata")
            graph_meta = prepare_graph_metadata(sql, sample_rows, state.user_query)
            print(f"[DATA_OBSERVATION] Graph metadata prepared: {graph_meta}")
            state.graph_data = graph_meta

        return state

    # If no data yet, call NL2SQL tool
    print("[DATA_OBSERVATION] Calling DB tool")
    tool_result = nl_to_sql_db_tool.invoke(state.user_query)

    if not tool_result.get("ok", False):
        state.data_ref = {
            "ok": False,
            "sql": tool_result.get("sql", ""),
            "error": tool_result.get("error", ""),
            "error_type": "data_retrieval"
        }
        return state

    state.data_ref = tool_result
    sql = tool_result["sql"]
    row_count = tool_result.get("row_count", 0)
    rows = tool_result.get("rows", [])
    sample_rows = tool_result.get("sample_rows", [])

    # Process based on size
    if row_count > 100:
        from tools import execute_aggregation_query
        agg_sql = generate_aggregation_query(sql, state.user_query)
        agg_result = execute_aggregation_query(agg_sql)
        
        if agg_result.get("ok"):
            agg_rows = agg_result.get("rows", [])
            state.data_ref["aggregation"] = agg_rows
            state.data_ref["message_type"] = "aggregated_data"
        else:
            state.data_ref["message_type"] = "sample_data"
    else:
        state.data_ref["message_type"] = "small_dataset"

    if state.need_graph:
        state.graph_data = prepare_graph_metadata(sql, sample_rows, state.user_query)

    return state


def generate_aggregation_query(original_sql: str, user_query: str) -> str:
    """
    Generates an aggregation query based on the original SQL.
    Uses LLM to create meaningful aggregations.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a SQL aggregation expert.

Given an original SQL query, create an aggregation query that provides:
- COUNT of records
- Statistical measures (MIN, MAX, AVG) for numeric columns
- GROUP BY date/time if temporal data exists

Rules:
- Output ONLY valid PostgreSQL SELECT query
- NO explanations, NO comments, NO markdown
- Use the same FROM clause as original
- Add meaningful aggregations
- Keep it concise (max 5-10 aggregation columns)
- End with semicolon

Example Input:
SELECT datetime, date, block, demand FROM lf.t_actual_demand WHERE date >= '2025-01-01'

Example Output:
SELECT DATE_TRUNC('day', date) as day, COUNT(*) as total_records, MIN(demand) as min_demand, MAX(demand) as max_demand, AVG(demand) as avg_demand FROM lf.t_actual_demand WHERE date >= '2025-01-01' GROUP BY DATE_TRUNC('day', date) ORDER BY day LIMIT 100;

CRITICAL: Output ONLY the SQL query. Nothing else.
"""
        ),
        ("user", "Original query:\n{sql}")
    ])

    try:
        messages = prompt.format_messages(sql=original_sql)
        response = llm.invoke(messages).content
        
        # Clean up response
        agg_sql = response.strip()
        
        if "SELECT" in agg_sql.upper():
            select_start = agg_sql.upper().find("SELECT")
            agg_sql = agg_sql[select_start:]
            
            if ";" in agg_sql:
                agg_sql = agg_sql[:agg_sql.find(";")+1]
            
            agg_sql = agg_sql.replace("```sql", "").replace("```", "").strip()
            
            lines = agg_sql.split('\n')
            sql_lines = []
            for line in lines:
                if any(word in line.lower() for word in ['this query', 'note that', 'also,', 'if you want']):
                    break
                sql_lines.append(line)
            
            agg_sql = ' '.join(sql_lines)
        
        agg_sql = ' '.join(agg_sql.split())
        
        if not agg_sql.endswith(';'):
            agg_sql += ';'
        
        print(f"[AGGREGATION] Cleaned SQL: {agg_sql}")
        return agg_sql
        
    except Exception as e:
        print(f"[AGGREGATION] Error generating query: {e}")
        return f"""
        SELECT COUNT(*) as total_records, AVG(demand) as avg_demand, MIN(demand) as min_demand, MAX(demand) as max_demand
        FROM ({original_sql.replace(';', '')}) as subquery;
        """

def prepare_graph_metadata(sql: str, *_args) -> dict:
    """
    Minimal graph metadata.
    Visualization logic is handled dynamically by graph_plotting_tool.
    """
    return {
        "sql": sql
    }


# -------------------------
# FORECASTING AGENT
# -------------------------
def forecasting_agent(state: GraphState) -> GraphState:
    """
    Handles demand forecasting queries.
    Maps to 'Forecasting AGENT' node in flowchart.
    """
    print("[FORECASTING] Processing forecast query")

    requested_date = extract_date_from_query(state.user_query)

    if not requested_date:
        state.data_ref = {
            "ok": False,
            "error_type": "missing_date",
            "message": "Please specify a date in YYYY-MM-DD format for forecasting."
        }
        return state

    dt = datetime.combine(requested_date, datetime.min.time())
    print(f"[FORECASTING] Requested date: {requested_date}")

    # Range validation
    if not is_within_range(dt, FORECAST_DATA_START, FORECAST_DATA_END):
        state.data_ref = {
            "ok": False,
            "error_type": "out_of_range",
            "message": build_out_of_range_message("forecast"),
            "requested_date": requested_date
        }
        state.is_out_of_range = True
        return state

    # If we have data from NL2SQL, use it
    if state.data_ref and state.data_ref.get("ok", True):
        rows = state.data_ref.get("rows", [])
        row_count = state.data_ref.get("row_count", 0)
        
        # Add forecast metadata
        state.data_ref["forecast_date"] = requested_date
        state.data_ref["message_type"] = "forecast_data"
        
        # Calculate statistics for summarization
        demands = [row.get('forecasted_demand', 0) for row in rows if 'forecasted_demand' in row]
        if demands:
            state.data_ref["statistics"] = {
                "avg_demand": sum(demands) / len(demands),
                "max_demand": max(demands),
                "min_demand": min(demands),
                "num_blocks": len(demands)
            }
        
        # Prepare graph if needed
        if state.need_graph:
            sql = state.data_ref.get("sql", "")
            sample_rows = state.data_ref.get("sample_rows", rows[:3])
            state.graph_data = prepare_graph_metadata(sql, sample_rows, state.user_query)
        
        return state

    # Otherwise, generate forecast query
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You must generate a SQL query to fetch forecasted demand data.

Template structure:
SELECT forecasted_demand, date, block
FROM lf.t_forecasted_demand
WHERE date = '<DATE>'
LIMIT 10;

Rules:
- Replace <DATE> with the requested date in YYYY-MM-DD format
- Use single quotes for dates
- Call the database tool with this query
"""
        ),
        ("user", state.user_query),
    ])

    llm_with_tools = llm.bind_tools([nl_to_sql_db_tool])
    response = llm_with_tools.invoke(prompt.format_messages())

    if not response.tool_calls:
        state.data_ref = {
            "ok": False,
            "error_type": "query_generation",
            "message": "Unable to generate forecast query."
        }
        return state

    tool_args = response.tool_calls[0]["args"]
    if isinstance(tool_args, dict):
        user_request = tool_args.get("user_request", state.user_query)
    else:
        user_request = tool_args

    tool_result = nl_to_sql_db_tool.invoke(user_request)

    if not tool_result.get("ok", False):
        state.data_ref = {
            "ok": False,
            "sql": tool_result.get("sql", ""),
            "error": tool_result.get("error", ""),
            "error_type": "forecast_retrieval"
        }
        return state

    state.data_ref = tool_result
    rows = tool_result.get("rows", [])
    
    # Add forecast metadata
    state.data_ref["forecast_date"] = requested_date
    state.data_ref["message_type"] = "forecast_data"
    
    # Calculate statistics
    demands = [row.get('forecasted_demand', 0) for row in rows if 'forecasted_demand' in row]
    if demands:
        state.data_ref["statistics"] = {
            "avg_demand": sum(demands) / len(demands),
            "max_demand": max(demands),
            "min_demand": min(demands),
            "num_blocks": len(demands)
        }

    return state


# -------------------------
# DECISION AND INTELLIGENCE AGENT
# -------------------------
def decision_intelligence_agent(state: GraphState) -> GraphState:
    """
    Provides business insights, recommendations, and decision support.
    Maps to 'Decision And Intelligence AGENT' node in flowchart.
    """
    print("[DECISION] Processing decision/intelligence query")

    # Use data if available
    context = ""
    if state.data_ref:
        rows = state.data_ref.get("rows", [])
        agg = state.data_ref.get("aggregation", [])
        
        # Provide cleaner context
        if agg:
            context = f"\n\nAvailable statistics:\n{agg[:5]}"
        elif rows:
            context = f"\n\nSample data ({len(rows)} records):\n{rows[:5]}"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a business intelligence agent for load forecasting.

Your role:
- Provide actionable insights
- Make data-driven recommendations
- Analyze trends and patterns
- Support decision-making

Guidelines:
- Be concise and actionable
- Use natural, conversational language
- Focus on business value
- Cite data when available
- Provide clear recommendations
- Use bullet points sparingly - prefer paragraphs
- Don't show technical details unless asked

Format your response in a friendly, professional way without excessive formatting.
"""
        ),
        ("user", "{query}{context}")
    ])

    response = llm.invoke(
        prompt.format_messages(
            query=state.user_query,
            context=context
        )
    ).content

    # Store the intelligence response for summarization
    state.data_ref = state.data_ref or {}
    state.data_ref["intelligence_response"] = response
    state.data_ref["message_type"] = "decision_intelligence"
    
    return state


# -------------------------
# UNIVERSAL SUMMARIZATION AGENT
# -------------------------
def summarization_agent(state: GraphState) -> GraphState:
    """
    Universal gateway for ALL responses to frontend.
    Processes data from all agents and creates human-readable outputs.
    Maps to 'Summarization AGENT' node in flowchart.
    """
    print("[SUMMARIZATION] Processing response for frontend delivery")
    print(f"[SUMMARIZATION] Intent: {state.intent}")
    print(f"[SUMMARIZATION] Has data_ref: {state.data_ref is not None}")
    print(f"[SUMMARIZATION] Need graph: {state.need_graph}")
    
    # First, handle graph plotting if needed
    if state.need_graph:
        state.graph_data = state.graph_data or {
            "sql": state.data_ref.get("sql")
        }
        state = execute_graph_plotting(state)

    # Check if there's an error to handle
    if state.data_ref and not state.data_ref.get("ok", True):
        state.final_answer = format_error_message(state)
        return state
    
    # Route to appropriate formatter based on intent and data type
    if state.intent == "text":
        # Text responses pass through or get formatted
        if not state.final_answer:
            state.final_answer = handle_text_response(state)
    
    elif state.intent == "data":
        state.final_answer = format_data_response(state)
    
    elif state.intent == "forecast":
        state.final_answer = format_forecast_response(state)
    
    elif state.intent == "decision":
        state.final_answer = format_decision_response(state)
    
    else:
        # Fallback
        state.final_answer = state.final_answer or "I've processed your request."
    
    print(f"[SUMMARIZATION] Final answer ready: {len(state.final_answer)} chars")
    return state


def execute_graph_plotting(state: GraphState) -> GraphState:
    """Execute graph plotting and add visualization info to response"""
    print("[SUMMARIZATION] ✓ Executing graph plotting")
    
    graph_metadata = state.graph_data
    sql = graph_metadata.get("sql")
    
    if not sql:
        print("[SUMMARIZATION] ERROR: No SQL in graph_data!")
        return state
    
    print(f"[SUMMARIZATION] Calling graph_plotting_tool with SQL: {sql[:80]}...")
    
    try:
        plot_result = graph_plotting_tool.invoke(sql)
        
        print(f"[SUMMARIZATION] Plot result: ok={plot_result.get('ok')}")
        
        if plot_result.get("ok"):
            print("[SUMMARIZATION] ✓✓✓ PLOT SUCCESS!")
            state.graph_data = plot_result
            state.graph_data["plot_success"] = True
        else:
            print(f"[SUMMARIZATION] Plot FAILED: {plot_result.get('error')}")
            state.graph_data["plot_success"] = False
            state.graph_data["plot_error"] = plot_result.get('error')
            
    except Exception as e:
        print(f"[SUMMARIZATION] EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        state.graph_data["plot_success"] = False
        state.graph_data["plot_error"] = str(e)
    
    return state


def format_error_message(state: GraphState) -> str:
    """Format error messages in user-friendly way"""
    data = state.data_ref
    error_type = data.get("error_type", "unknown")
    show_tech = should_show_technical_details(state.user_query)
    
    if error_type == "missing_date":
        return data.get("message", "Please specify a date for forecasting.")
    
    elif error_type == "out_of_range":
        return data.get("message", "The requested date is outside the available range.")
    
    elif error_type in ["sql_execution", "data_retrieval", "forecast_retrieval"]:
        if show_tech:
            return (
                "⚠️ I couldn't retrieve the data.\n\n"
                f"**SQL Query:**\n```sql\n{data.get('sql', 'N/A')}\n```\n\n"
                f"**Error:**\n{data.get('error', 'Unknown error')}"
            )
        else:
            return (
                "⚠️ I couldn't retrieve the data from the database.\n\n"
                "💡 *Tip: Ask me to 'show the SQL query' if you want technical details.*"
            )
    
    else:
        return "⚠️ An error occurred while processing your request."


def handle_text_response(state: GraphState) -> str:
    """Handle text/general query responses"""
    if state.final_answer:
        return state.final_answer
    
    # Generate a friendly response
    return "I'm here to help with load forecasting queries. What would you like to know?"


def format_data_response(state: GraphState) -> str:
    """Format data query responses using LLM for natural presentation"""
    data = state.data_ref or {}
    message_type = data.get("message_type", "unknown")
    
    if message_type == "no_data":
        return "I couldn't find any records matching your query in the database."
    
    # Prepare data summary for LLM
    row_count = data.get("row_count", 0)
    rows = data.get("rows", [])
    agg = data.get("aggregation", [])
    sql = data.get("sql", "")
    
    show_tech = should_show_technical_details(state.user_query)
    
    # Build context for LLM
    context = {
        "user_query": state.user_query,
        "row_count": row_count,
        "has_aggregation": bool(agg),
        "aggregation_data": agg[:3] if agg else [],
        "sample_rows": rows[:5] if rows else [],
        "show_technical": show_tech
    }
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a data presentation specialist for a load forecasting system.

Your task: Convert raw database results into natural, human-readable responses.

Guidelines:
1. Write in natural, conversational language
2. Lead with the key insights
3. Use clear formatting (emojis are fine: 📊 📈 ⚡)
4. For aggregated data: Highlight patterns, trends, statistics
5. For small datasets: Present key findings, not raw tables
6. Be concise but informative
7. Don't show SQL queries unless show_technical is True
8. Focus on what the user asked for

Data types:
- If has_aggregation: Focus on statistics and trends
- If sample_rows: Present the actual data in a clean way
- Always mention total row count if large (>10)

Examples:
User asks: "What was demand on 2025-01-15?"
Good: "On January 15, 2025, electricity demand averaged 850.5 MW, with peak demand reaching 1,200 MW around midday."
Bad: "I found 96 records. The SQL query returned..."

User asks: "Show me demand trends for January"
Good: "📊 January showed 2,976 demand records with average demand of 825 MW. Peak periods consistently occurred between 6-9 PM with demands exceeding 1,100 MW."
Bad: "Here are the first 10 rows of data..."
"""
        ),
        ("user", """
User query: {user_query}

Data summary:
- Total records: {row_count}
- Has aggregation: {has_aggregation}
- Aggregation data: {aggregation_data}
- Sample rows: {sample_rows}
- Show technical details: {show_technical}

Create a natural, human-readable response to the user's query.
""")
    ])
    
    try:
        messages = prompt.format_messages(
            user_query=state.user_query,
            row_count=row_count,
            has_aggregation=context["has_aggregation"],
            aggregation_data=str(context["aggregation_data"]),
            sample_rows=str(context["sample_rows"]),
            show_technical=show_tech
        )
        
        response = llm.invoke(messages).content
        
        # Add technical details if requested
        if show_tech:
            response += f"\n\n**Technical Details:**\n```sql\n{sql}\n```\n"
            response += f"*Returned {row_count:,} records*"
        
        # Add graph info if available
        if state.graph_data and state.graph_data.get("plot_success"):
            plot_result = state.graph_data
            response += (
                        "\n\n📈 **Visualization Created**\n"
                        "A dynamic chart was generated based on your query."
                    )

        elif state.graph_data and not state.graph_data.get("plot_success"):
            response += f"\n\n⚠️ Note: Visualization could not be created: {state.graph_data.get('plot_error', 'Unknown error')}"
        
        return response
        
    except Exception as e:
        print(f"[SUMMARIZATION] LLM formatting error: {e}")
        # Fallback to basic formatting
        return format_data_basic(data, show_tech)


def format_data_basic(data: dict, show_tech: bool) -> str:
    """Fallback basic data formatting"""
    row_count = data.get("row_count", 0)
    agg = data.get("aggregation", [])
    rows = data.get("rows", [])
    
    if agg:
        first_agg = dict(agg[0])
        if 'avg_demand' in first_agg:
            return (
                f"📊 Found {row_count:,} records.\n\n"
                f"Average demand: {first_agg.get('avg_demand', 0):.1f} MW\n"
                f"Peak demand: {first_agg.get('max_demand', 0):.1f} MW\n"
                f"Minimum demand: {first_agg.get('min_demand', 0):.1f} MW"
            )
    
    return f"📊 Found {row_count:,} records matching your query."


def format_forecast_response(state: GraphState) -> str:
    """Format forecast responses using LLM"""
    data = state.data_ref or {}
    forecast_date = data.get("forecast_date")
    statistics = data.get("statistics", {})
    rows = data.get("rows", [])
    
    show_tech = should_show_technical_details(state.user_query)
    
    if not rows:
        return f"No forecast data available for {forecast_date}."
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a forecast presentation specialist.

Your task: Present demand forecast data in a clear, actionable way.

Guidelines:
1. Lead with the date and key forecast
2. Highlight: average demand, peak demand, minimum demand
3. Mention number of time blocks covered
4. Use natural language, avoid jargon
5. Be concise but informative
6. Use emojis sparingly (🔮 for forecast, 📊 for stats)

Example:
"🔮 Forecast for January 15, 2025

Expected electricity demand will average 850 MW throughout the day, with peak demand of 1,200 MW expected during evening hours. The forecast covers 96 time blocks, showing demand ranging from 600 MW in early morning to peak afternoon levels."
"""
        ),
        ("user", """
Forecast date: {forecast_date}
Statistics:
- Average demand: {avg_demand:.1f} MW
- Peak demand: {max_demand:.1f} MW
- Minimum demand: {min_demand:.1f} MW
- Time blocks: {num_blocks}

User query: {user_query}

Create a natural forecast summary.
""")
    ])
    
    try:
        messages = prompt.format_messages(
            forecast_date=forecast_date,
            avg_demand=statistics.get("avg_demand", 0),
            max_demand=statistics.get("max_demand", 0),
            min_demand=statistics.get("min_demand", 0),
            num_blocks=statistics.get("num_blocks", 0),
            user_query=state.user_query
        )
        
        response = llm.invoke(messages).content
        
        # Add technical details if requested
        if show_tech:
            sql = data.get("sql", "")
            response += f"\n\n**Technical Details:**\n```sql\n{sql}\n```"
        
        # Add graph info if available
        if state.graph_data and state.graph_data.get("plot_success"):
            plot_result = state.graph_data
            response += (
                f"\n\n📈 **Visualization Created**\n"
                f"I've generated a {plot_result['plot_type']} chart showing the forecast trend."
            )
        
        return response
        
    except Exception as e:
        print(f"[SUMMARIZATION] Forecast formatting error: {e}")
        # Fallback
        return (
            f"🔮 **Forecast for {forecast_date}**\n\n"
            f"Expected electricity demand:\n"
            f"• Average: {statistics.get('avg_demand', 0):.1f} MW\n"
            f"• Peak: {statistics.get('max_demand', 0):.1f} MW\n"
            f"• Minimum: {statistics.get('min_demand', 0):.1f} MW\n\n"
            f"📊 Forecast covers {statistics.get('num_blocks', 0)} time blocks"
        )


def format_decision_response(state: GraphState) -> str:
    """Format decision intelligence responses"""
    data = state.data_ref or {}
    intelligence_response = data.get("intelligence_response", "")
    
    if intelligence_response:
        response = intelligence_response
    else:
        response = "I've analyzed the available data. What specific insights would you like?"
    
    # Add graph info if available
    if state.graph_data and state.graph_data.get("plot_success"):
        plot_result = state.graph_data
        response += (
            f"\n\n📈 **Visualization Created**\n"
            f"I've generated a {plot_result['plot_type']} chart to support this analysis."
        )
    
    return response


# -------------------------
# TEXT AGENT (for general queries)
# -------------------------
def text_agent(state: GraphState) -> GraphState:
    """
    Handles general text queries that don't require tools.
    """
    print("[TEXT] Handling general query")
    
    # Handle simple greetings directly
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    query_lower = state.user_query.lower().strip()
    
    if query_lower in greetings:
        state.final_answer = (
            "Hello! 👋 I'm your Load Forecasting Assistant.\n\n"
            "I can help you with:\n"
            "• Historical demand data and trends\n"
            "• Demand forecasting and predictions\n"
            "• Holiday impact analysis\n"
            "• Performance metrics\n"
            "• Business insights and recommendations\n"
            "• Data visualizations and charts\n\n"
            "What would you like to know?"
        )
        return state
    
    # For other queries, use LLM
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a helpful assistant for a load forecasting system.

Provide clear, accurate answers about:
- Load forecasting concepts
- System capabilities
- General domain knowledge
- Explanations of terms and methods

Be conversational, friendly, and concise.
Use natural language without excessive formatting.
Avoid bullet points unless listing specific items.
"""
        ),
        ("user", "{query}")
    ])

    response = llm.invoke(
        prompt.format_messages(query=state.user_query)
    ).content

    state.final_answer = response
    return state
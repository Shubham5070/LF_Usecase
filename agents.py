from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, date
import re
import json

from state import GraphState
from tools import nl_to_sql_db_tool, execute_db_query, graph_plotting_tool, cleanup_temp_table
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
# QUERY ANALYSIS AGENT (Entry Point)
# -------------------------
def query_analysis_agent(state: GraphState) -> GraphState:
    """
    Analyzes user query and determines routing.
    Maps to 'Query analysis' node in flowchart.
    """
    print("[QUERY_ANALYSIS] Analyzing user query")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a query analysis agent for a load forecasting system.

Your task:
- Classify the user's intent
- Determine required tools and agents

Intent categories:
- "data" - Historical data queries (actual demand, holidays, metrics)
- "forecast" - Future demand predictions
- "decision" - Business intelligence, recommendations, insights
- "text" - General questions, explanations, definitions

Tool requirements:
- need_db_call: true if query needs database access
- need_graph: true ONLY if user explicitly asks for: trend, plot, graph, chart, visualize, show variation
- need_model_run: true if user wants to train/run a new model
- need_api_call: true if external API data is needed

Rules:
- Forecast queries ALWAYS need DB
- Data queries about actual/historical records need DB
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
    
    # Planner validates and refines the plan
    # Already done by query_analysis_agent, so this can be a pass-through
    # or add additional validation logic here
    
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

    # Directly call the tool with the user query
    # The tool itself handles NL to SQL conversion
    print(f"[NL2SQL] Calling DB tool with query: {state.user_query}")
    
    tool_result = nl_to_sql_db_tool.invoke(state.user_query)

    if not tool_result.get("ok", False):
        state.final_answer = (
            "⚠️ SQL execution failed.\n\n"
            f"SQL:\n{tool_result.get('sql')}\n\n"
            f"Error:\n{tool_result.get('error')}"
        )
        return state

    # Store results
    state.data_ref = tool_result["data_ref"]
    state.data_ref["sql"] = tool_result["sql"]

    print(f"[NL2SQL] Success: {state.data_ref['row_count']} rows retrieved")
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

    # If we already have data from NL2SQL, format it
    if state.data_ref:
        sql = state.data_ref.get("sql", "")
        row_count = state.data_ref.get("row_count", 0)
        table_name = state.data_ref.get("name", "")

        if row_count == 0:
            state.final_answer = (
                "🧾 Generated SQL:\n"
                f"{sql}\n\n"
                "📊 No records found in the database."
            )
            return state

        print(f"[DATA_OBSERVATION] Dataset has {row_count} rows")

        # For large datasets (>100 rows), generate aggregation query
        if row_count > 100:
            print("[DATA_OBSERVATION] Large dataset detected - generating aggregation")
            
            # Generate aggregation SQL
            agg_sql = generate_aggregation_query(sql, state.user_query)
            print(f"[DATA_OBSERVATION] Aggregation SQL: {agg_sql}")
            
            # Execute aggregation
            agg_result = execute_db_query(agg_sql)
            
            if agg_result.get("ok"):
                agg_rows = agg_result["data_ref"].get("rows", [])
                
                state.final_answer = (
                    "✅ Data retrieved successfully.\n\n"
                    "🧾 Original Query:\n"
                    f"{sql}\n\n"
                    "📊 Dataset Summary:\n"
                    f"- Total rows: {row_count:,}\n"
                    f"- Source table: {table_name}\n\n"
                    "📈 Aggregated Statistics:\n"
                    f"{format_aggregation_results(agg_rows)}"
                )
            else:
                # Fallback to sample if aggregation fails
                state.final_answer = format_sample_data(state.data_ref, sql)
        else:
            # Small dataset - show directly
            state.final_answer = format_sample_data(state.data_ref, sql)

        # Prepare graph data if needed
        if state.need_graph:
            print("[DATA_OBSERVATION] Preparing graph metadata")
            state.graph_data = prepare_graph_metadata(table_name, sql, state.user_query)

        return state

    # If no data yet, call NL2SQL tool
    print("[DATA_OBSERVATION] Calling DB tool")
    tool_result = nl_to_sql_db_tool.invoke(state.user_query)

    if not tool_result.get("ok", False):
        state.final_answer = (
            "⚠️ Data retrieval failed.\n\n"
            f"SQL:\n{tool_result.get('sql')}\n\n"
            f"Error:\n{tool_result.get('error')}"
        )
        return state

    state.data_ref = tool_result["data_ref"]
    sql = tool_result["sql"]
    row_count = state.data_ref.get("row_count", 0)

    # Process based on size
    if row_count > 100:
        agg_sql = generate_aggregation_query(sql, state.user_query)
        agg_result = execute_db_query(agg_sql)
        
        if agg_result.get("ok"):
            agg_rows = agg_result["data_ref"].get("rows", [])
            state.final_answer = (
                "✅ Data retrieved successfully.\n\n"
                f"📊 Total rows: {row_count:,}\n\n"
                "📈 Statistics:\n"
                f"{format_aggregation_results(agg_rows)}"
            )
        else:
            state.final_answer = format_sample_data(state.data_ref, sql)
    else:
        state.final_answer = format_sample_data(state.data_ref, sql)

    if state.need_graph:
        table_name = state.data_ref.get("name", "")
        state.graph_data = prepare_graph_metadata(table_name, sql, state.user_query)

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
- Top N records if categorical data exists

Rules:
- Output ONLY valid PostgreSQL SELECT query
- Use the same FROM clause as original
- Add meaningful aggregations
- Keep it concise (max 5-10 aggregation columns)
- End with semicolon

Example:
Original: SELECT demand, date FROM lf.t_actual_demand WHERE date >= '2025-01-01'
Aggregated: 
SELECT 
    COUNT(*) as total_records,
    MIN(demand) as min_demand,
    MAX(demand) as max_demand,
    AVG(demand) as avg_demand,
    DATE_TRUNC('day', date) as day,
    COUNT(*) as records_per_day
FROM lf.t_actual_demand 
WHERE date >= '2025-01-01'
GROUP BY DATE_TRUNC('day', date)
ORDER BY day
LIMIT 100;
"""
        ),
        ("user", "Original query:\n{sql}\n\nUser question: {query}")
    ])

    try:
        messages = prompt.format_messages(sql=original_sql, query=user_query)
        response = llm.invoke(messages).content
        
        # Clean up response
        agg_sql = response.replace("```sql", "").replace("```", "").strip()
        
        return agg_sql
    except Exception as e:
        print(f"[AGGREGATION] Error generating query: {e}")
        # Fallback to simple aggregation
        return f"""
        SELECT COUNT(*) as total_records
        FROM ({original_sql.replace(';', '')}) as subquery;
        """


def format_aggregation_results(rows: list) -> str:
    """
    Formats aggregation results in a readable way.
    """
    if not rows:
        return "No aggregation data available"
    
    result_lines = []
    for row in rows[:20]:  # Limit to 20 aggregated rows
        row_dict = dict(row)
        formatted = []
        for key, value in row_dict.items():
            if isinstance(value, (int, float)):
                formatted.append(f"{key}: {value:,.2f}")
            else:
                formatted.append(f"{key}: {value}")
        result_lines.append(" | ".join(formatted))
    
    return "\n".join(result_lines)


def format_sample_data(data_ref: dict, sql: str) -> str:
    """
    Formats sample data for small datasets.
    """
    rows = data_ref.get("rows", [])
    row_count = data_ref.get("row_count", 0)
    
    records_text = "\n".join(str(dict(row)) for row in rows[:10])
    
    return (
        "✅ Data retrieved successfully.\n\n"
        "🧾 Generated SQL:\n"
        f"{sql}\n\n"
        "📊 Summary:\n"
        f"- Total rows: {row_count}\n\n"
        "📄 Sample Records:\n"
        f"{records_text}"
    )


def prepare_graph_metadata(table_name: str, sql: str, user_query: str) -> dict:
    """
    Prepares metadata for graph plotting without fetching all data yet.
    Returns column information and SQL for later data fetching.
    """
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
Analyze the SQL query and user request to determine:
1. What should be on X-axis (typically date/time)
2. What should be on Y-axis (typically the metric being measured)
3. What type of plot (line, bar, scatter)

Respond ONLY in JSON format with these exact fields:
- x_column: the column name for x-axis
- y_column: the column name for y-axis
- plot_type: either "line", "bar", or "scatter"
- title: a descriptive title for the plot

Example response:
{{"x_column": "date", "y_column": "demand", "plot_type": "line", "title": "Demand Trend"}}
"""
        ),
        ("user", "SQL: {sql}\n\nUser request: {query}")
    ])
    
    try:
        # Format the prompt with variables
        messages = prompt.format_messages(sql=sql, query=user_query)
        response = llm.invoke(messages).content
        
        # Clean up response - remove markdown code blocks if present
        response = response.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        metadata = json.loads(response)
        metadata["table_name"] = table_name
        metadata["sql"] = sql
        return metadata
    except Exception as e:
        print(f"[GRAPH_METADATA] Parse error: {e}, using defaults")
        return {
            "table_name": table_name,
            "sql": sql,
            "x_column": "datetime",
            "y_column": "demand",
            "plot_type": "line",
            "title": "Data Visualization"
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
        state.final_answer = "Please specify a date in YYYY-MM-DD format for forecasting."
        return state

    dt = datetime.combine(requested_date, datetime.min.time())
    print(f"[FORECASTING] Requested date: {requested_date}")

    # Range validation
    if not is_within_range(dt, FORECAST_DATA_START, FORECAST_DATA_END):
        state.final_answer = build_out_of_range_message("forecast")
        state.is_out_of_range = True
        return state

    # If we have data from NL2SQL, use it
    if state.data_ref:
        sql = state.data_ref.get("sql", "")
        rows = state.data_ref.get("rows", [])
        row_count = state.data_ref.get("row_count", 0)
        
        records_text = "\n".join(str(dict(row)) for row in rows[:10])

        state.final_answer = (
            "✅ Forecast retrieved successfully.\n\n"
            "🧾 Generated SQL:\n"
            f"{sql}\n\n"
            "📊 Summary:\n"
            f"- Total rows: {row_count}\n\n"
            "📄 Forecast Records:\n"
            f"{records_text}"
        )
        
        # Prepare graph if needed
        if state.need_graph:
            table_name = state.data_ref.get("name", "")
            state.graph_data = prepare_graph_metadata(table_name, sql, state.user_query)
        
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
        state.final_answer = "Unable to generate forecast query."
        return state

    # Extract tool arguments properly
    tool_args = response.tool_calls[0]["args"]
    if isinstance(tool_args, dict):
        user_request = tool_args.get("user_request", state.user_query)
    else:
        user_request = tool_args

    tool_result = nl_to_sql_db_tool.invoke(user_request)

    if not tool_result.get("ok", False):
        state.final_answer = (
            "⚠️ Forecast retrieval failed.\n\n"
            f"SQL:\n{tool_result.get('sql')}\n\n"
            f"Error:\n{tool_result.get('error')}"
        )
        return state

    state.data_ref = tool_result["data_ref"]
    sql = tool_result["sql"]
    rows = state.data_ref.get("rows", [])

    records_text = "\n".join(str(dict(row)) for row in rows[:10])

    state.final_answer = (
        "✅ Forecast retrieved successfully.\n\n"
        "🧾 Generated SQL:\n"
        f"{sql}\n\n"
        "📊 Summary:\n"
        f"- Total rows: {state.data_ref['row_count']}\n\n"
        "📄 Forecast Records:\n"
        f"{records_text}"
    )

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
        context = f"\n\nAvailable data:\n{rows}"

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
- Focus on business value
- Cite data when available
- Provide clear recommendations
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

    state.final_answer = response
    return state


# -------------------------
# SUMMARIZATION AGENT
# -------------------------
def summarization_agent(state: GraphState) -> GraphState:
    """
    Summarizes and presents final results.
    Maps to 'Summarization AGENT' node in flowchart.
    """
    print("[SUMMARIZATION] Generating final summary")

    # Execute graph plotting if needed
    if state.need_graph and state.graph_data:
        print("[SUMMARIZATION] Executing graph plotting")
        
        graph_metadata = state.graph_data
        table_name = graph_metadata.get("table_name")
        
        if table_name:
            # Call graph plotting tool with full data
            plot_result = graph_plotting_tool.invoke({
                "table_name": table_name,
                "x_column": graph_metadata.get("x_column", "date"),
                "y_column": graph_metadata.get("y_column", "demand"),
                "plot_type": graph_metadata.get("plot_type", "line"),
                "title": graph_metadata.get("title", "Data Trend"),
                "limit": 10000  # Fetch up to 10k points for plotting
            })
            
            if plot_result.get("ok"):
                print(f"[SUMMARIZATION] Plot saved: {plot_result['filepath']}")
                
                # Add plot info to final answer
                plot_info = (
                    f"\n\n📊 **Visualization Generated**\n"
                    f"- Type: {plot_result['plot_type']}\n"
                    f"- Data points: {plot_result['data_points']:,}\n"
                    f"- Saved to: `{plot_result['filepath']}`\n"
                    f"- Filename: `{plot_result['filename']}`"
                )
                
                if state.final_answer:
                    state.final_answer += plot_info
                else:
                    state.final_answer = f"✅ Graph created successfully!{plot_info}"
                
                # Store plot result in state
                state.graph_data = plot_result
            else:
                error_msg = f"\n\n⚠️ Graph generation failed: {plot_result.get('error')}"
                if state.final_answer:
                    state.final_answer += error_msg
                else:
                    state.final_answer = error_msg

    # Clean up temporary table if exists
    if state.data_ref and state.data_ref.get("name"):
        table_name = state.data_ref["name"]
        print(f"[SUMMARIZATION] Cleaning up temp table: {table_name}")
        cleanup_temp_table(table_name)

    # If it's a text-only response (no data), skip enhancement
    if state.intent == "text" and state.final_answer:
        print("[SUMMARIZATION] Text response - passing through")
        return state

    # If final_answer already exists, enhance it with LLM
    if state.final_answer:
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are a summarization agent for a load forecasting system.

Your task:
- Present information clearly and professionally
- Highlight key insights and patterns
- Make technical information accessible
- Keep visualization references intact

Guidelines:
- Preserve all file paths, statistics, and technical details
- Add context and interpretation where helpful
- Use emojis sparingly for readability
- Keep the tone professional but friendly

IMPORTANT: Do not remove or modify:
- File paths (e.g., plots/...)
- Statistics (row counts, metrics)
- SQL queries
- Data tables
"""
            ),
            ("user", "Enhance this response for clarity:\n\n{answer}")
        ])

        try:
            enhanced = llm.invoke(
                prompt.format_messages(answer=state.final_answer)
            ).content
            
            state.final_answer = enhanced
        except Exception as e:
            print(f"[SUMMARIZATION] Enhancement failed: {e}, using original")
            # Keep original answer if enhancement fails

    return state


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
            "• Historical demand data queries\n"
            "• Demand forecasting predictions\n"
            "• Holiday information\n"
            "• Performance metrics analysis\n"
            "• Business insights and recommendations\n"
            "• Data visualizations\n\n"
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
"""
        ),
        ("user", "{query}")
    ])

    response = llm.invoke(
        prompt.format_messages(query=state.user_query)
    ).content

    state.final_answer = response
    return state
# agents.py
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, date
import re
import json

from state import GraphState
from tools import nl_to_sql_db_tool, graph_plotting_tool
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
    print(f"[NL2SQL] Calling DB tool with query: {state.user_query}")
    
    tool_result = nl_to_sql_db_tool.invoke(state.user_query)

    if not tool_result.get("ok", False):
        state.final_answer = (
            "⚠️ SQL execution failed.\n\n"
            f"SQL:\n{tool_result.get('sql')}\n\n"
            f"Error:\n{tool_result.get('error')}"
        )
        return state

    # Store results directly (no data_ref wrapper anymore)
    state.data_ref = tool_result  # The whole result is now the data_ref
    
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

    # If we already have data from NL2SQL, format it
    if state.data_ref:
        sql = state.data_ref.get("sql", "")
        row_count = state.data_ref.get("row_count", 0)
        rows = state.data_ref.get("rows", [])
        sample_rows = state.data_ref.get("sample_rows", [])

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
            
            # Execute aggregation using execute_aggregation_query
            from tools import execute_aggregation_query
            agg_result = execute_aggregation_query(agg_sql)
            
            if agg_result.get("ok"):
                agg_rows = agg_result.get("rows", [])
                
                state.final_answer = (
                    "✅ Data retrieved successfully.\n\n"
                    "🧾 Original Query:\n"
                    f"{sql}\n\n"
                    "📊 Dataset Summary:\n"
                    f"- Total rows: {row_count:,}\n\n"
                    "📈 Aggregated Statistics:\n"
                    f"{format_aggregation_results(agg_rows)}"
                )
            else:
                # Fallback to sample if aggregation fails
                state.final_answer = format_sample_data(rows, sql, row_count)
        else:
            # Small dataset - show directly
            state.final_answer = format_sample_data(rows, sql, row_count)

        # Prepare graph data if needed (use only sample_rows for metadata)
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
        state.final_answer = (
            "⚠️ Data retrieval failed.\n\n"
            f"SQL:\n{tool_result.get('sql')}\n\n"
            f"Error:\n{tool_result.get('error')}"
        )
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
            state.final_answer = (
                "✅ Data retrieved successfully.\n\n"
                f"📊 Total rows: {row_count:,}\n\n"
                "📈 Statistics:\n"
                f"{format_aggregation_results(agg_rows)}"
            )
        else:
            state.final_answer = format_sample_data(rows, sql, row_count)
    else:
        state.final_answer = format_sample_data(rows, sql, row_count)

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
        
        # Clean up response - remove any markdown, explanations, etc.
        agg_sql = response.strip()
        
        # Extract just the SQL if there's extra text
        # Look for SELECT statement
        if "SELECT" in agg_sql.upper():
            # Find the SELECT statement
            select_start = agg_sql.upper().find("SELECT")
            agg_sql = agg_sql[select_start:]
            
            # Find the end (semicolon or end of useful SQL)
            if ";" in agg_sql:
                agg_sql = agg_sql[:agg_sql.find(";")+1]
            
            # Remove markdown
            agg_sql = agg_sql.replace("```sql", "").replace("```", "").strip()
            
            # Remove any text after the query
            lines = agg_sql.split('\n')
            sql_lines = []
            for line in lines:
                # Stop at lines that look like explanations
                if any(word in line.lower() for word in ['this query', 'note that', 'also,', 'if you want']):
                    break
                sql_lines.append(line)
            
            agg_sql = ' '.join(sql_lines)
        
        # Final cleanup
        agg_sql = ' '.join(agg_sql.split())  # Normalize whitespace
        
        if not agg_sql.endswith(';'):
            agg_sql += ';'
        
        print(f"[AGGREGATION] Cleaned SQL: {agg_sql}")
        
        return agg_sql
    except Exception as e:
        print(f"[AGGREGATION] Error generating query: {e}")
        # Fallback to simple aggregation
        return f"""
        SELECT COUNT(*) as total_records, AVG(demand) as avg_demand, MIN(demand) as min_demand, MAX(demand) as max_demand
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


def format_sample_data(rows: list, sql: str, row_count: int) -> str:
    """
    Formats sample data for small datasets.
    """
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


def prepare_graph_metadata(sql: str, sample_rows: list, user_query: str) -> dict:
    """
    Prepares metadata for graph plotting using only 3 sample rows.
    Returns SQL query and column info for later plotting.
    """
    
    print(f"[GRAPH_METADATA] Preparing with {len(sample_rows)} sample rows")
    
    # If no sample rows, try to detect from SQL
    if not sample_rows:
        print("[GRAPH_METADATA] No sample rows, using defaults")
        return {
            "sql": sql,
            "x_column": "datetime",
            "y_column": "demand",
            "plot_type": "line",
            "title": "Actual Demand Trend"
        }
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
Analyze the SQL query and sample data to determine:
1. What should be on X-axis (typically date/time)
2. What should be on Y-axis (typically the metric being measured)
3. What type of plot (line, bar, scatter)

Respond ONLY in JSON format with these exact fields:
- x_column: the column name for x-axis
- y_column: the column name for y-axis
- plot_type: either "line", "bar", or "scatter"
- title: a descriptive title for the plot

NO explanations, NO markdown, ONLY JSON.

Example:
{{"x_column": "day", "y_column": "avg_demand", "plot_type": "line", "title": "Demand Trend"}}
"""
        ),
        ("user", "SQL: {sql}\n\nSample data columns: {columns}\n\nUser request: {query}")
    ])
    
    try:
        # Get column names from sample data
        columns = list(sample_rows[0].keys()) if sample_rows else []
        
        # Format the prompt with variables
        messages = prompt.format_messages(
            sql=sql, 
            columns=columns,
            query=user_query
        )
        response = llm.invoke(messages).content
        
        print(f"[GRAPH_METADATA] LLM response: {response}")
        
        # Clean up response
        response = response.replace("```json", "").replace("```", "").strip()
        
        # Parse JSON
        metadata = json.loads(response)
        metadata["sql"] = sql  # Store SQL for re-execution during plotting
        
        print(f"[GRAPH_METADATA] Parsed metadata: {metadata}")
        return metadata
        
    except Exception as e:
        print(f"[GRAPH_METADATA] Parse error: {e}, using defaults")
        
        # Try to detect columns from sample data
        if sample_rows:
            columns = list(sample_rows[0].keys())
            print(f"[GRAPH_METADATA] Available columns: {columns}")
            
            # Smart column detection
            x_col = None
            y_col = None
            
            # Look for time-based columns for x-axis
            for col in columns:
                if any(word in col.lower() for word in ['date', 'time', 'day', 'month']):
                    x_col = col
                    break
            
            # Look for numeric columns for y-axis
            for col in columns:
                if any(word in col.lower() for word in ['demand', 'value', 'avg', 'sum', 'count']):
                    y_col = col
                    break
            
            # Fallback to first two columns
            if not x_col:
                x_col = columns[0] if columns else "date"
            if not y_col:
                y_col = columns[1] if len(columns) > 1 else "demand"
            
            print(f"[GRAPH_METADATA] Detected columns: x={x_col}, y={y_col}")
        else:
            x_col = "datetime"
            y_col = "demand"
            
        return {
            "sql": sql,
            "x_column": x_col,
            "y_column": y_col,
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

    state.data_ref = tool_result
    sql = tool_result["sql"]
    rows = tool_result.get("rows", [])
    row_count = tool_result.get("row_count", 0)

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
# SUMMARIZATION AGENT - FIXED VERSION
# -------------------------
def summarization_agent(state: GraphState) -> GraphState:
    """
    Summarizes and presents final results.
    Maps to 'Summarization AGENT' node in flowchart.
    """
    print("[SUMMARIZATION] Generating final summary")
    print(f"[SUMMARIZATION] need_graph={state.need_graph}, has_graph_data={state.graph_data is not None}")

    # Execute graph plotting if needed - FIXED TO USE SQL NOT TABLE_NAME
    if state.need_graph and state.graph_data:
        print("[SUMMARIZATION] ✓ Executing graph plotting")
        print(f"[SUMMARIZATION] Graph metadata: {state.graph_data}")
        
        graph_metadata = state.graph_data
        sql = graph_metadata.get("sql")  # CHANGED FROM table_name TO sql
        
        print(f"[SUMMARIZATION] SQL extracted: {sql is not None}")
        
        if sql:  # CHANGED FROM if table_name TO if sql
            print(f"[SUMMARIZATION] Calling graph_plotting_tool with SQL: {sql[:80]}...")
            
            try:
                plot_result = graph_plotting_tool.invoke({
                    "sql": sql,  # CHANGED FROM table_name TO sql
                    "x_column": graph_metadata.get("x_column", "date"),
                    "y_column": graph_metadata.get("y_column", "demand"),
                    "plot_type": graph_metadata.get("plot_type", "line"),
                    "title": graph_metadata.get("title", "Data Trend"),
                    "limit": 10000
                })
                
                print(f"[SUMMARIZATION] Plot result: ok={plot_result.get('ok')}")
                
                if plot_result.get("ok"):
                    print(f"[SUMMARIZATION] ✓✓✓ PLOT SUCCESS! File: {plot_result['filepath']}")
                    
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
                    
                    state.graph_data = plot_result
                else:
                    print(f"[SUMMARIZATION] Plot FAILED: {plot_result.get('error')}")
                    error_msg = f"\n\n⚠️ Graph generation failed: {plot_result.get('error')}"
                    if state.final_answer:
                        state.final_answer += error_msg
                        
            except Exception as e:
                print(f"[SUMMARIZATION] EXCEPTION: {e}")
                import traceback
                traceback.print_exc()
                error_msg = f"\n\n⚠️ Graph exception: {str(e)}"
                if state.final_answer:
                    state.final_answer += error_msg
        else:
            print("[SUMMARIZATION] ERROR: No SQL in graph_data!")
    else:
        print(f"[SUMMARIZATION] Skipping graph - need_graph={state.need_graph}, has_data={state.graph_data is not None}")

    # If it's a text-only response, skip enhancement
    if state.intent == "text" and state.final_answer:
        print("[SUMMARIZATION] Text response - passing through")
        return state

    # Simple pass-through, no LLM enhancement
    if state.final_answer:
        print("[SUMMARIZATION] Response ready")

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
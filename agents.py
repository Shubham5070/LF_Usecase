from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, date
import re
import json
from state import GraphState
from tools import nl_to_sql_db_tool, graph_plotting_tool, execute_query, model_run_tool
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
# LLM (lazy-initialized to avoid heavy imports at module-import time)
# -------------------------
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm

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
   
    from datetime import datetime
    import re
    # current_date = datetime.now().strftime("%Y-%m-%d")
    current_date = "2026-01-14"  # Fixed date for consistent testing
    current_date_readable = datetime.now().strftime("%B %d, %Y")
   
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a query analysis agent for a load forecasting system.
 
TODAY'S DATE: {current_date} ({current_date_readable})
 
DATABASE SCHEMA:
- t_actual_demand: Actual/real historical demand
- t_forecasted_demand: Past/future forecasted/predicted demand values (historical predictions)
- t_holidays: Holiday information
- t_metrics: Model performance metrics
 
═══════════════════════════════════════════════════════════════
⚠️  HIGHEST PRIORITY RULE - DATE/TIME BASED ROUTING ⚠️
═══════════════════════════════════════════════════════════════
ROUTING RULES:
 
1. GRAPH CHECK (Priority 1):
   - Keywords: plot, show, graph, chart, visualize, visual, display
   - If graph → need_graph: true, need_db_call: true
   - Graph without time → intent: "data"
   - Graph with time → apply date routing
   
STEP 1: IDENTIFY IF QUERY HAS A DATE/TIME REFERENCE
- Look for: specific dates, "tomorrow", "next week", "last month", "yesterday", etc.
 
STEP 2: DETERMINE IF DATE IS PAST OR FUTURE
- Compare mentioned date with TODAY ({current_date})
- If date is BEFORE or EQUAL to {current_date} → PAST
- If date is AFTER {current_date} → FUTURE
 
STEP 3: ROUTE BASED ON TIME
- PAST date/time → intent: "data" (even if query mentions "forecast" or "predicted")
- FUTURE date/time → intent: "forecast"
 
═══════════════════════════════════════════════════════════════
 
EXAMPLES WITH TODAY = {current_date}:
 
Query: "give forecasted demand of 25 june 2025"
→ Date: 2025-06-25
→ Comparison: 2025-06-25 is BEFORE {current_date}
→ Result: intent: "data", need_db_call: true, need_model_run: false
 
Query: "give forecasted demand for tomorrow"
→ Date: {current_date} + 1 day = FUTURE
→ Result: intent: "forecast", need_db_call: true, need_model_run: true
 
Query: "what was the forecasted demand on 15 january 2025"
→ Date: 2025-01-15
→ Comparison: 2025-01-15 is BEFORE {current_date}
→ Result: intent: "data", need_db_call: true, need_model_run: false
 
Query: "predict demand for next month"
→ Date: February 2026 = FUTURE
→ Result: intent: "forecast", need_db_call: true, need_model_run: true
 
Query: "show me actual demand for last week"
→ Time: last week = PAST
→ Result: intent: "data", need_db_call: true, need_model_run: false
 
Query: "forecast for 30 january 2026"
→ Date: 2026-01-30
→ Comparison: 2026-01-30 is AFTER {current_date}
→ Result: intent: "forecast", need_db_call: true, need_model_run: true
 
═══════════════════════════════════════════════════════════════
 
INTENT CLASSIFICATION (AFTER TIME-BASED ROUTING):
 
1. "data" intent → ALL past/historical queries:
   - Any query with dates BEFORE or EQUAL TO {current_date}
   - Queries about actual demand from the past
   - Queries about forecasted demand from the past (stored in t_forecasted_demand)
   - Queries about holidays, metrics
   - Time keywords: was, were, last, previous, historical, past, yesterday, [any past date]
 
2. "forecast" intent → ONLY future predictions:
   - Any query with dates AFTER {current_date}
   - Queries requesting new predictions for future dates
   - Time keywords: will, tomorrow, next, upcoming, future, predict for [future date]
 
3. "decision" intent → Business intelligence (can be past or future context):
   - Comparisons, recommendations, insights
   - Multi-source analysis
   - Keywords: compare, recommend, suggest, optimize, should, strategy, what-if
   - Time-based routing still applies if specific dates mentioned
 
4. "text" intent → General information (no date context):
   - Explanations, definitions, concepts
   - No database needed
   - Keywords: what is, explain, how does, define
 
═══════════════════════════════════════════════════════════════
 
TOOL REQUIREMENTS:
 
need_db_call:
- true for "data" intent (accessing past data from all tables)
- true for "forecast" intent (may need historical data + generating predictions)
- true for "decision" intent (needs data for analysis)
- false for "text" intent
 
need_graph:
- true ONLY if explicitly requested: plot, graph, chart, visualize, show trend, visual
- false otherwise
 
need_model_run:
- true ONLY for "forecast" intent (generating NEW predictions for FUTURE dates)
- false for "data" intent (querying EXISTING past data)
- false for "decision" intent (unless explicitly asking to run model)
 
need_api_call:
- true if external data needed (weather, events for future predictions)
- false otherwise
 
═══════════════════════════════════════════════════════════════
 
CRITICAL REMINDERS:
1. ⚠️  DATE/TIME COMPARISON IS THE FIRST AND MOST IMPORTANT STEP
2. ⚠️  "June 25, 2025" is BEFORE today ({current_date}), so it's PAST → "data" intent
3. ⚠️  Even if query says "forecast" or "predict", if date is PAST → "data" intent
4. ⚠️  Only route to "forecast" if date is genuinely in the FUTURE
 
═══════════════════════════════════════════════════════════════
 
CRITICAL OUTPUT FORMAT - READ CAREFULLY:
 
YOU MUST OUTPUT ONLY THE JSON OBJECT BELOW. NO OTHER TEXT.
NO explanations. NO markdown. NO code blocks. NO extra words.
JUST the JSON starting with {{ and ending with }}.
 
{{"intent": "data", "need_db_call": true, "need_graph": true, "need_model_run": false, "need_api_call": false}}
"""
        ),
        ("user", "{query}")
    ])
   
    response = _get_llm().invoke(
        prompt.format_messages(
            query=state.user_query,
            current_date=current_date,
            current_date_readable=current_date_readable
        )
    ).content
   
    print(f"[QUERY_ANALYSIS] Raw response (length={len(response)}): {repr(response[:500])}")
   
    # Parse JSON response with maximum robustness
    try:
        if not response or not response.strip():
            raise ValueError("Empty response from LLM")
       
        # Clean the response
        response_cleaned = response.strip()
       
        # Remove any BOM or invisible characters at the start
        response_cleaned = response_cleaned.lstrip('\ufeff\u200b\u200c\u200d')
       
        # Remove markdown code blocks
        if "```json" in response_cleaned:
            match = re.search(r'```json\s*(\{.*?\})\s*```', response_cleaned, re.DOTALL)
            if match:
                response_cleaned = match.group(1).strip()
        elif "```" in response_cleaned:
            match = re.search(r'```\s*(\{.*?\})\s*```', response_cleaned, re.DOTALL)
            if match:
                response_cleaned = match.group(1).strip()
       
        # Extract JSON object using regex if there's extra text
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_cleaned, re.DOTALL)
        if json_match:
            response_cleaned = json_match.group(0)
       
        # Remove any trailing/leading whitespace and newlines
        response_cleaned = response_cleaned.strip()
       
        print(f"[QUERY_ANALYSIS] Cleaned response: {repr(response_cleaned[:200])}")
       
        # Try to parse JSON
        result = json.loads(response_cleaned)
       
        # Validate required fields
        required_fields = ["intent", "need_db_call", "need_graph", "need_model_run", "need_api_call"]
        missing_fields = [field for field in required_fields if field not in result]
       
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
       
        # Validate intent values
        valid_intents = ["data", "forecast", "decision", "text"]
        if result["intent"] not in valid_intents:
            print(f"[QUERY_ANALYSIS] ⚠️  Invalid intent '{result['intent']}', defaulting to 'text'")
            result["intent"] = "text"
       
        # Set state values
        state.intent = result["intent"]
        state.need_db_call = bool(result["need_db_call"])
        state.need_graph = bool(result["need_graph"])
        state.need_model_run = bool(result["need_model_run"])
        state.need_api_call = bool(result["need_api_call"])
       
        print("[QUERY_ANALYSIS] ✅ Successfully parsed JSON response")
       
    except json.JSONDecodeError as e:
        print(f"[QUERY_ANALYSIS] ❌ JSON Parse error: {e}")
        print(f"[QUERY_ANALYSIS] Failed at position {e.pos}")
        print(f"[QUERY_ANALYSIS] Cleaned response bytes: {response_cleaned.encode('utf-8')[:100]}")
       
        # Try to extract intent manually as last resort
        intent_match = re.search(r'"intent"\s*:\s*"(\w+)"', response)
        if intent_match:
            state.intent = intent_match.group(1)
            print(f"[QUERY_ANALYSIS] 🔧 Extracted intent manually: {state.intent}")
        else:
            state.intent = "text"
       
        # Use fallback values
        state.need_db_call = False
        state.need_graph = False
        state.need_model_run = False
        state.need_api_call = False
       
    except ValueError as e:
        print(f"[QUERY_ANALYSIS] ❌ Validation error: {e}")
        print(f"[QUERY_ANALYSIS] Raw response: {repr(response[:300])}")
        state.intent = "text"
        state.need_db_call = False
        state.need_graph = False
        state.need_model_run = False
        state.need_api_call = False
       
    except Exception as e:
        print(f"[QUERY_ANALYSIS] ❌ Unexpected error: {type(e).__name__}: {e}")
        print(f"[QUERY_ANALYSIS] Raw response: {repr(response[:300])}")
        state.intent = "text"
        state.need_db_call = False
        state.need_graph = False
        state.need_model_run = False
        state.need_api_call = False
   
    print(
        f"[QUERY_ANALYSIS] Result → intent={state.intent}, "
        f"need_db_call={state.need_db_call}, "
        f"need_graph={state.need_graph}, "
        f"need_model_run={state.need_model_run}, "
        f"need_api_call={state.need_api_call}"
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
        response = _get_llm().invoke(messages).content
        
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
    # IMPORTANT: allow planner-forced model runs even when the requested date
    # is outside the recorded forecast availability. Only block when the
    # planner DID NOT request a model run.
    if not state.need_model_run and not is_within_range(dt, FORECAST_DATA_START, FORECAST_DATA_END):
        state.data_ref = {
            "ok": False,
            "error_type": "out_of_range",
            "message": build_out_of_range_message("forecast"),
            "requested_date": requested_date
        }
        state.is_out_of_range = True
        return state

    # If the planner requested a model run, ALWAYS run the model and use ONLY the
    # predictions produced by that run. Do NOT use pre-existing DB rows even if
    # they exist for the same date — the planner's request takes precedence.
    if state.need_model_run:
        print(f"[FORECASTING] need_model_run=True → forcing model run for {requested_date} (ignoring existing DB predictions)")

        # Force a fresh run (run_and_store_forecast deletes prior rows for the same date)
        run_result = model_run_tool.invoke(str(requested_date))

        if not run_result.get("ok"):
            state.data_ref = {
                "ok": False,
                "error_type": "model_run_failure",
                "message": run_result.get("error", "Model run failed")
            }
            return state

        # Fetch only the rows for the requested date (these should reflect the run we just performed)
        fetch_sql = (
            "SELECT datetime, block, predicted_demand AS forecasted_demand, model_id, generated_at "
            "FROM t_predicted_demand_chatbot "
            f"WHERE prediction_date = '{requested_date}' "
            "ORDER BY datetime"
        )

        fetch_res = execute_query(fetch_sql)
        if not fetch_res.get("ok"):
            state.data_ref = {
                "ok": False,
                "error_type": "forecast_fetch_failure",
                "message": fetch_res.get("error", "Failed to fetch predictions after model run")
            }
            return state

        if fetch_res.get("row_count", 0) == 0:
            state.data_ref = {
                "ok": False,
                "error_type": "no_predictions_after_run",
                "message": "Model run completed but no predictions were saved to the database."
            }
            return state

        # Return ONLY the freshly-generated rows and attach run metadata
        state.data_ref = {
            "ok": True,
            "sql": fetch_sql,
            "rows": fetch_res.get("rows", []),
            "row_count": fetch_res.get("row_count", 0),
            "sample_rows": fetch_res.get("rows", [])[:3],
            "generated_by_run": True,
            "run_metrics": run_result.get("metrics")
        }

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

    llm_with_tools = _get_llm().bind_tools([nl_to_sql_db_tool])
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

    response = _get_llm().invoke(
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
        
        response = _get_llm().invoke(messages).content
        
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
        
        response = _get_llm().invoke(messages).content
        
        # Add technical details if requested
        if show_tech:
            sql = data.get("sql", "")
            response += f"\n\n**Technical Details:**\n```sql\n{sql}\n```"

        # If the forecast was produced by a fresh run but no validation metrics
        # are available (expected for forward-looking forecasts), add a short
        # clarification instead of attempting to show metrics.
        if data.get("generated_by_run") and not data.get("run_metrics"):
            response += "\n\n⚠️ No validation metrics available for forward-looking forecasts (actuals not yet observed)."

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

    response = _get_llm().invoke(
        prompt.format_messages(query=state.user_query)
    ).content

    state.final_answer = response
    return state
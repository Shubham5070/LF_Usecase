# agent.py - MERGED VERSION with improved graph functionality

from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, date
import re
import json
from state import GraphState
from tools import nl_to_sql_db_tool, graph_plotting_tool, execute_query, model_run_tool
from llm import get_llm
from agent_llm_config import get_agent_llm, get_agent_model_name
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
import os
DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()

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


# Helper: stage-aware printing that can optionally include the LLM model name
def _env_bool(name: str, default: bool) -> bool:
    v = __import__("os").environ.get(name)
    if v is None:
        return default
    return str(v).lower() not in ("0", "false", "no")


def _model_name_from_llm(llm) -> str | None:
    """Try common attributes to extract a concise model name from an LLM instance."""
    if not llm:
        return None
    for attr in ("model", "model_name", "name"):
        val = getattr(llm, attr, None)
        if val:
            return str(val)
    # best-effort from repr
    try:
        s = str(llm)
        import re
        m = re.search(r"model=['\"]?([^'\")]+)", s)
        if m:
            return m.group(1)
        return s
    except Exception:
        return None


def stage_print(stage: str, msg: str, *, llm=None, model_name: str | None = None) -> None:
    """Print to stdout with an optional [STAGE][MODEL] tag controlled by env vars.

    Prefer an explicit `model_name` (reads from config) to avoid instantiating LLMs.

    Env vars:
      LOG_SHOW_LLM_MODEL (bool, default: true) - globally enable/disable model tags
      LOG_SHOW_LLM_MODEL_STAGES (comma list) - if set, only show model for these stages (e.g. SUMMARIZATION)
    """
    import os
    stage_up = stage.upper()
    show_models = _env_bool("LOG_SHOW_LLM_MODEL", True)
    stages_raw = os.environ.get("LOG_SHOW_LLM_MODEL_STAGES", "")
    stages_set = {s.strip().upper() for s in stages_raw.split(",") if s.strip()} if stages_raw else set()

    model_tag = ""
    if show_models and (not stages_set or stage_up in stages_set):
        # explicit model_name takes precedence
        resolved = model_name or (_model_name_from_llm(llm) if llm is not None else None)
        if not resolved:
            # best-effort: try agent config (non-instantiating) if available
            try:
                from agent_llm_config import get_agent_model_name as _cfg_model
                resolved = _cfg_model(stage.lower())
            except Exception:
                resolved = None
        if not resolved:
            # last resort: global getter
            try:
                from llm import get_llm as _getllm
                resolved = _model_name_from_llm(_getllm())
            except Exception:
                resolved = None
        if resolved:
            model_tag = f"[{str(resolved).split(':')[0].upper()}]"

    print(f"[{stage_up}]{model_tag} {msg}")

# -------------------------
# QUERY ANALYSIS AGENT (Entry Point)
# -------------------------



def query_analysis_agent(state: GraphState) -> GraphState:
    """
    Analyzes user query and determines routing using hybrid approach.
    Maps to 'Query analysis' node in flowchart.
   
    Approach:
    1. Try rule-based classification first (fast, deterministic)
    2. Fall back to LLM for ambiguous cases
    """
    print("[QUERY_ANALYSIS] Analyzing user query")
   
    from datetime import datetime, timedelta
    import re
   
    query = state.user_query.lower().strip()
    current_date = datetime.now()
    current_date_str = current_date.strftime("%Y-%m-%d")
    current_date_readable = current_date.strftime("%B %d, %Y")
   
    # ═══════════════════════════════════════════════════════════
    # PHASE 1: RULE-BASED CLASSIFICATION
    # ═══════════════════════════════════════════════════════════
   
    rule_result = _apply_rule_based_classification(query, current_date)
   
    if rule_result["confident"]:
        # Rule-based classification is confident
        print(f"[QUERY_ANALYSIS] ✅ RULE-BASED classification (confidence: {rule_result['confidence_reason']})")
        state.intent = rule_result["intent"]
        state.need_db_call = rule_result["need_db_call"]
        state.need_graph = rule_result["need_graph"]
        state.need_model_run = rule_result["need_model_run"]
        state.need_api_call = rule_result["need_api_call"]
       
        print(
            f"[QUERY_ANALYSIS] Result → intent={state.intent}, "
            f"need_db_call={state.need_db_call}, "
            f"need_graph={state.need_graph}, "
            f"need_model_run={state.need_model_run}, "
            f"need_api_call={state.need_api_call}"
        )
        return state
   
    # ═══════════════════════════════════════════════════════════
    # PHASE 2: LLM-BASED CLASSIFICATION (for ambiguous cases)
    # ═══════════════════════════════════════════════════════════
   
    print(f"[QUERY_ANALYSIS] 🤖 LLM-BASED classification (reason: {rule_result['confidence_reason']})")
   
    llm = get_agent_llm("query_analysis")
   
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
ROUTING RULES:
 
1. GRAPH CHECK:
   - Keywords: plot, show, graph, chart, visualize, visual, display
   - If graph → need_graph: true, need_db_call: true
   
2. DATE/TIME BASED ROUTING:
   - Compare mentioned date with TODAY ({current_date})
   - If date is BEFORE or EQUAL to {current_date} → PAST → intent: "data"
   - If date is AFTER {current_date} → FUTURE → intent: "forecast"
 
INTENT CLASSIFICATION:
 
1. "data" → ALL past/historical queries
2. "forecast" → ONLY future predictions  
3. "decision" → Business intelligence/comparisons
4. "text" → General information (no database needed)
 
TOOL REQUIREMENTS:
- need_db_call: true for data/forecast/decision, false for text
- need_graph: true ONLY if explicitly requested
- need_model_run: true ONLY for "forecast" intent
- need_api_call: true if external data needed
 
═══════════════════════════════════════════════════════════════
 
OUTPUT FORMAT: Return ONLY valid JSON, no other text.
 
{{"intent": "data", "need_db_call": true, "need_graph": true, "need_model_run": false, "need_api_call": false}}
"""
        ),
        ("user", "{query}")
    ])
   
    response = llm.invoke(
        prompt.format_messages(
            query=state.user_query,
            current_date=current_date_str,
            current_date_readable=current_date_readable
        )
    ).content
   
    print(f"[QUERY_ANALYSIS] Raw LLM response (length={len(response)}): {repr(response[:300])}")
   
    # Parse LLM response
    try:
        response_cleaned = _extract_json_from_response(response)
        print(f"[QUERY_ANALYSIS] Cleaned response: {repr(response_cleaned[:200])}")
       
        result = json.loads(response_cleaned)
       
        # Validate
        required_fields = ["intent", "need_db_call", "need_graph", "need_model_run", "need_api_call"]
        if not all(field in result for field in required_fields):
            raise ValueError(f"Missing required fields")
       
        valid_intents = ["data", "forecast", "decision", "text"]
        if result["intent"] not in valid_intents:
            result["intent"] = "text"
       
        state.intent = result["intent"]
        state.need_db_call = bool(result["need_db_call"])
        state.need_graph = bool(result["need_graph"])
        state.need_model_run = bool(result["need_model_run"])
        state.need_api_call = bool(result["need_api_call"])
       
        print("[QUERY_ANALYSIS] ✅ Successfully parsed LLM response")
       
    except Exception as e:
        print(f"[QUERY_ANALYSIS] ❌ LLM parsing failed: {e}")
        print(f"[QUERY_ANALYSIS] Falling back to rule-based result")
       
        # Use rule-based result as fallback even if not confident
        state.intent = rule_result["intent"]
        state.need_db_call = rule_result["need_db_call"]
        state.need_graph = rule_result["need_graph"]
        state.need_model_run = rule_result["need_model_run"]
        state.need_api_call = rule_result["need_api_call"]
   
    print(
        f"[QUERY_ANALYSIS] Result → intent={state.intent}, "
        f"need_db_call={state.need_db_call}, "
        f"need_graph={state.need_graph}, "
        f"need_model_run={state.need_model_run}, "
        f"need_api_call={state.need_api_call}"
    )
    return state
 
 
def _apply_rule_based_classification(query: str, current_date: datetime) -> dict:
    """
    Apply rule-based classification logic.
    Returns classification with confidence indicator.
    """
    import re
    from datetime import timedelta
   
    result = {
        "intent": "text",
        "need_db_call": False,
        "need_graph": False,
        "need_model_run": False,
        "need_api_call": False,
        "confident": False,
        "confidence_reason": ""
    }
   
    # ═══════════════════════════════════════════════════════════
    # RULE 1: CHECK FOR GRAPH/VISUALIZATION REQUEST
    # ═══════════════════════════════════════════════════════════
    graph_keywords = r'\b(plot|graph|chart|visuali[sz]e|show.*trend|display.*graph|draw)\b'
    if re.search(graph_keywords, query):
        result["need_graph"] = True
        result["need_db_call"] = True
   
    # ═══════════════════════════════════════════════════════════
    # RULE 2: METRIC QUERIES (HIGH PRIORITY - BEFORE GENERAL QUERIES)
    # ═══════════════════════════════════════════════════════════
    metric_keywords = r'\b(ape|mape|rmse|mae|mse|r2|accuracy|error|metric|performance|score)\b'
    if re.search(metric_keywords, query):
        result["intent"] = "data"
        result["need_db_call"] = True
        result["confident"] = True
        result["confidence_reason"] = "Metric/performance query detected"
   
    # ═══════════════════════════════════════════════════════════
    # RULE 3: DATE EXTRACTION AND TIME-BASED ROUTING (MULTI-LIBRARY)
    # ═══════════════════════════════════════════════════════════
    date_info = _extract_date_from_query_multi(query, current_date)
   
    if date_info and date_info['date']:
        extracted_date = date_info['date']
        is_future = extracted_date > current_date
       
        # Format both dates in same format
        current_date_formatted = current_date.strftime("%Y-%m-%d")
        extracted_date_formatted = extracted_date.strftime("%Y-%m-%d")
       
        # Print detailed comparison
        print(f"\n{'='*60}")
        print(f"[DATE_COMPARISON] Current Date:    {current_date_formatted}")
        print(f"[DATE_COMPARISON] Extracted Date:  {extracted_date_formatted}")
        print(f"[DATE_COMPARISON] Extracted By:    {date_info['library']}")
        print(f"[DATE_COMPARISON] Original Format: {date_info['original_format']}")
        print(f"[DATE_COMPARISON] Consensus:       {date_info['consensus']}")
        print(f"[DATE_COMPARISON] Is Future?       {is_future}")
        print(f"{'='*60}\n")
       
        if is_future:
            result["intent"] = "forecast"
            result["need_db_call"] = True
            result["need_model_run"] = True
            result["confident"] = True
            result["confidence_reason"] = f"Future date: {extracted_date_formatted} (by {date_info['library']})"
        else:
            result["intent"] = "data"
            result["need_db_call"] = True
            result["need_model_run"] = False
            result["confident"] = True
            result["confidence_reason"] = f"Past date: {extracted_date_formatted} (by {date_info['library']})"
       
        return result
   
    # If we detected metrics but no specific date, still return as data query
    if result["intent"] == "data" and result["confident"]:
        return result
   
    # ═══════════════════════════════════════════════════════════
    # RULE 4: GENERAL/DEFINITION QUERIES
    # ═══════════════════════════════════════════════════════════
    general_patterns = [
        r'^\s*what\s+is\s+',
        r'^\s*how\s+does\s+',
        r'^\s*explain\s+',
        r'^\s*define\s+',
        r'^\s*tell\s+me\s+about\s+',
    ]
   
    for pattern in general_patterns:
        if re.search(pattern, query):
            # Check if it's NOT asking about specific data/metrics
            if not re.search(r'\b(demand|forecast|actual|predict|data|value|number|ape|mape|rmse|mae|metric|error|accuracy|holiday|performance)\b', query):
                result["intent"] = "text"
                result["confident"] = True
                result["confidence_reason"] = "General information query"
                return result
   
    # ═══════════════════════════════════════════════════════════
    # RULE 5: RELATIVE TIME KEYWORDS
    # ═══════════════════════════════════════════════════════════
   
    # PAST indicators
    past_keywords = r'\b(yesterday|last\s+(week|month|year|quarter)|previous|historical|past|was|were)\b'
    if re.search(past_keywords, query):
        result["intent"] = "data"
        result["need_db_call"] = True
        result["need_model_run"] = False
        result["confident"] = True
        result["confidence_reason"] = "Past time keyword detected"
        return result
   
    # FUTURE indicators
    future_keywords = r'\b(tomorrow|next\s+(week|month|year|quarter)|upcoming|future|will\s+be|predict|forecast)\b'
    if re.search(future_keywords, query):
        result["intent"] = "forecast"
        result["need_db_call"] = True
        result["need_model_run"] = True
        result["confident"] = True
        result["confidence_reason"] = "Future time keyword detected"
        return result
   
    # ═══════════════════════════════════════════════════════════
    # RULE 6: BUSINESS INTELLIGENCE QUERIES
    # ═══════════════════════════════════════════════════════════
    decision_keywords = r'\b(compare|comparison|recommend|suggest|should|optimize|strategy|best|worst|what.if|versus|vs\.?)\b'
    if re.search(decision_keywords, query):
        result["intent"] = "decision"
        result["need_db_call"] = True
        result["confident"] = True
        result["confidence_reason"] = "Decision/comparison keyword detected"
        return result
   
    # ═══════════════════════════════════════════════════════════
    # RULE 7: DATA QUERIES (MEDIUM CONFIDENCE)
    # ═══════════════════════════════════════════════════════════
    data_keywords = r'\b(actual|real|historical|recorded|measured|holiday|metric|performance)\b'
    if re.search(data_keywords, query):
        result["intent"] = "data"
        result["need_db_call"] = True
        result["confident"] = True
        result["confidence_reason"] = "Data-specific keyword detected"
        return result
   
    # ═══════════════════════════════════════════════════════════
    # NO CLEAR PATTERN - LOW CONFIDENCE
    # ═══════════════════════════════════════════════════════════
    result["intent"] = "text"
    result["confident"] = False
    result["confidence_reason"] = "Ambiguous query, needs LLM analysis"
    return result
 
 
# ═══════════════════════════════════════════════════════════
# MULTI-LIBRARY DATE EXTRACTION (WITHOUT DUCKLING)
# ═══════════════════════════════════════════════════════════
 
def _extract_date_from_query_multi(query: str, current_date: datetime) -> dict:
    """
    Extract date using 3 libraries with consensus voting.
    Uses: dateparser, dateutil, parsedatetime
   
    Returns:
        dict with date, library name, original format, and consensus info
        or None if no date found
    """
    from datetime import timedelta
    from collections import Counter
    import re
   
    print(f"[DATE_EXTRACT] Analyzing query: '{query}'")
   
    dates_found = []
   
    # ═══════════════════════════════════════════════════════════
    # METHOD 1: dateparser (most comprehensive)
    # ═══════════════════════════════════════════════════════════
    try:
        import dateparser
       
        settings = {
            'RELATIVE_BASE': current_date,
            'PREFER_DATES_FROM': 'future',
            'RETURN_AS_TIMEZONE_AWARE': False,
        }
       
        parsed = dateparser.parse(query, settings=settings)
        if parsed:
            dates_found.append({
                'library': 'dateparser',
                'date': parsed,
                'confidence': 'high'
            })
            print(f"[DATE_EXTRACT] ✓ dateparser    → {parsed.strftime('%Y-%m-%d')}")
    except Exception as e:
        print(f"[DATE_EXTRACT] ✗ dateparser    → Failed: {e}")
   
    # ═══════════════════════════════════════════════════════════
    # METHOD 2: dateutil (robust parsing)
    # ═══════════════════════════════════════════════════════════
    try:
        from dateutil import parser as dateutil_parser
       
        # Extract potential date strings
        date_candidates = re.findall(
            r'\b\d{1,4}[-/]\d{1,2}[-/]\d{1,4}\b|'
            r'\b(?:tomorrow|yesterday|today)\b|'
            r'\b(?:next|last)\s+(?:week|month|year)\b|'
            r'\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b',
            query,
            re.IGNORECASE
        )
       
        for candidate in date_candidates:
            try:
                parsed = dateutil_parser.parse(candidate, default=current_date, fuzzy=True)
                dates_found.append({
                    'library': 'dateutil',
                    'date': parsed,
                    'confidence': 'medium',
                    'matched_text': candidate
                })
                print(f"[DATE_EXTRACT] ✓ dateutil     → {parsed.strftime('%Y-%m-%d')} (from '{candidate}')")
                break
            except:
                continue
               
        if not any(d['library'] == 'dateutil' for d in dates_found):
            print(f"[DATE_EXTRACT] ✗ dateutil     → No valid date found")
           
    except Exception as e:
        print(f"[DATE_EXTRACT] ✗ dateutil     → Failed: {e}")
   
    # ═══════════════════════════════════════════════════════════
    # METHOD 3: parsedatetime (natural language)
    # ═══════════════════════════════════════════════════════════
    try:
        import parsedatetime as pdt
       
        cal = pdt.Calendar()
        time_struct, parse_status = cal.parse(query, sourceTime=current_date.timetuple())
       
        # parse_status: 1 = date, 2 = time, 3 = datetime
        if parse_status in [1, 3]:
            parsed = datetime(*time_struct[:6])
            dates_found.append({
                'library': 'parsedatetime',
                'date': parsed,
                'confidence': 'medium'
            })
            print(f"[DATE_EXTRACT] ✓ parsedatetime → {parsed.strftime('%Y-%m-%d')}")
        else:
            print(f"[DATE_EXTRACT] ✗ parsedatetime → No valid date found (status: {parse_status})")
           
    except Exception as e:
        print(f"[DATE_EXTRACT] ✗ parsedatetime → Failed: {e}")
   
    # ═══════════════════════════════════════════════════════════
    # CONSENSUS VOTING
    # ═══════════════════════════════════════════════════════════
   
    if not dates_found:
        print("[DATE_EXTRACT] ❌ No dates found by any library, trying regex fallback...")
        regex_date = _extract_date_from_query_regex(query, current_date)
        if regex_date:
            return {
                'date': regex_date,
                'library': 'regex',
                'original_format': 'pattern-matched',
                'consensus': '1/1 (regex only)'
            }
        return None
   
    # Normalize dates to date-only (ignore time)
    normalized_dates = []
    for entry in dates_found:
        date_only = entry['date'].date()
        normalized_dates.append({
            'date': date_only,
            'library': entry['library'],
            'confidence': entry['confidence'],
            'matched_text': entry.get('matched_text', '')
        })
   
    # Count occurrences
    date_counter = Counter([entry['date'] for entry in normalized_dates])
    most_common_date, count = date_counter.most_common(1)[0]
   
    total_parsers = len(dates_found)
    consensus_ratio = count / total_parsers
   
    print(f"\n[DATE_EXTRACT] Consensus Analysis:")
    print(f"  - Most common date: {most_common_date}")
    print(f"  - Agreement: {count}/{total_parsers} parsers ({consensus_ratio*100:.0f}%)")
   
    # Find which library provided the consensus date
    winning_entry = next(e for e in normalized_dates if e['date'] == most_common_date)
   
    # Determine original format from query
    original_format = _detect_date_format(query, most_common_date)
   
    # Require at least 50% consensus
    if consensus_ratio >= 0.5:
        result_date = datetime.combine(most_common_date, datetime.min.time())
        print(f"[DATE_EXTRACT] ✅ CONSENSUS REACHED: {result_date.strftime('%Y-%m-%d')}\n")
       
        return {
            'date': result_date,
            'library': winning_entry['library'],
            'original_format': original_format,
            'consensus': f"{count}/{total_parsers} parsers agree"
        }
   
    # If no consensus, use high-confidence result
    high_conf_dates = [e for e in normalized_dates if e['confidence'] == 'high']
    if high_conf_dates:
        result_date = datetime.combine(high_conf_dates[0]['date'], datetime.min.time())
        print(f"[DATE_EXTRACT] ✅ Using high-confidence: {result_date.strftime('%Y-%m-%d')}\n")
       
        return {
            'date': result_date,
            'library': high_conf_dates[0]['library'],
            'original_format': original_format,
            'consensus': f"high-confidence ({high_conf_dates[0]['library']})"
        }
   
    # Fallback: first result
    result_date = datetime.combine(normalized_dates[0]['date'], datetime.min.time())
    print(f"[DATE_EXTRACT] ⚠️  Using first result: {result_date.strftime('%Y-%m-%d')}\n")
   
    return {
        'date': result_date,
        'library': normalized_dates[0]['library'],
        'original_format': original_format,
        'consensus': f"1/{total_parsers} (fallback)"
    }
 
 
def _detect_date_format(query: str, date_obj) -> str:
    """
    Detect the original format of the date in the query.
    """
    import re
   
    # Common patterns with their format descriptions
    patterns = [
        (r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b', 'DD/MM/YYYY or MM/DD/YYYY'),
        (r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', 'YYYY-MM-DD'),
        (r'\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}\b', 'DD Month YYYY'),
        (r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', 'Month DD, YYYY'),
        (r'\btomorrow\b', 'relative: tomorrow'),
        (r'\byesterday\b', 'relative: yesterday'),
        (r'\btoday\b', 'relative: today'),
        (r'\bnext\s+week\b', 'relative: next week'),
        (r'\blast\s+week\b', 'relative: last week'),
        (r'\bnext\s+month\b', 'relative: next month'),
        (r'\blast\s+month\b', 'relative: last month'),
    ]
   
    for pattern, format_name in patterns:
        if re.search(pattern, query, re.IGNORECASE):
            return format_name
   
    return 'natural language'
 
 
def _extract_date_from_query_regex(query: str, current_date: datetime) -> datetime:
    """
    Fallback regex-based date extraction.
    """
    import re
    from datetime import timedelta
   
    print("[DATE_EXTRACT] Using regex fallback...")
   
    # Relative dates
    if 'tomorrow' in query:
        print("[DATE_EXTRACT] ✓ Regex → tomorrow")
        return current_date + timedelta(days=1)
    if 'yesterday' in query:
        print("[DATE_EXTRACT] ✓ Regex → yesterday")
        return current_date - timedelta(days=1)
    if re.search(r'\btoday\b', query):
        print("[DATE_EXTRACT] ✓ Regex → today")
        return current_date
   
    if re.search(r'\bnext\s+week\b', query):
        print("[DATE_EXTRACT] ✓ Regex → next week")
        return current_date + timedelta(weeks=1)
    if re.search(r'\bnext\s+month\b', query):
        print("[DATE_EXTRACT] ✓ Regex → next month")
        return current_date + timedelta(days=30)
    if re.search(r'\blast\s+week\b', query):
        print("[DATE_EXTRACT] ✓ Regex → last week")
        return current_date - timedelta(weeks=1)
    if re.search(r'\blast\s+month\b', query):
        print("[DATE_EXTRACT] ✓ Regex → last month")
        return current_date - timedelta(days=30)
   
    # Specific date patterns
    date_patterns = [
        (r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b',
         lambda m: datetime(int(m.group(3)), int(m.group(2)), int(m.group(1)))),
        (r'\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b',
         lambda m: datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))),
        (r'\b(\d{1,2})\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b',
         lambda m: datetime(int(m.group(3)), _month_to_num(m.group(2)), int(m.group(1)))),
        (r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})\b',
         lambda m: datetime(int(m.group(3)), _month_to_num(m.group(1)), int(m.group(2)))),
    ]
   
    for pattern, converter in date_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            try:
                result = converter(match)
                print(f"[DATE_EXTRACT] ✓ Regex → {result.strftime('%Y-%m-%d')}")
                return result
            except:
                continue
   
    print("[DATE_EXTRACT] ✗ Regex → No match found")
    return None
 
 
def _month_to_num(month_name: str) -> int:
    """Convert month name to number."""
    months = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4,
        'may': 5, 'june': 6, 'july': 7, 'august': 8,
        'september': 9, 'october': 10, 'november': 11, 'december': 12
    }
    return months.get(month_name.lower(), 1)
 
 
def _extract_json_from_response(response: str) -> str:
    """Extract JSON from LLM response."""
    import re
   
    if not response or not response.strip():
        raise ValueError("Empty response")
   
    response_cleaned = response.strip()
    response_cleaned = response_cleaned.lstrip('\ufeff\u200b\u200c\u200d')
   
    if "```json" in response_cleaned:
        match = re.search(r'```json\s*(\{.*?\})\s*```', response_cleaned, re.DOTALL)
        if match:
            return match.group(1).strip()
    elif "```" in response_cleaned:
        match = re.search(r'```\s*(\{.*?\})\s*```', response_cleaned, re.DOTALL)
        if match:
            return match.group(1).strip()
   
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_cleaned, re.DOTALL)
    if json_match:
        return json_match.group(0).strip()
   
    return response_cleaned.strip()
 
 
 
 




# ------------------------------------------------
#-------------------------------------------------

def get_season_and_block_context(date_val, rows: list) -> dict:
    """
    Derive season label(s) and block-period breakdown from rows.
    
    CORRECTED VERSION - Fixes:
    1. Accurate Indian seasons with Post-Monsoon period
    2. Handles multi-date/multi-month data safely
    3. Robust block parsing with type checking
    4. Safe datetime parsing with error handling
    5. Handles None values gracefully
    6. Optimized performance for large datasets
    """
    from datetime import datetime
    
    # -----------------------------
    # SEASON LOGIC (CORRECTED)
    # -----------------------------
    def month_to_season(month: int) -> str:
        """
        Convert month number to Indian season.
        Indian climate seasons:
        - Winter: Dec, Jan, Feb (12, 1, 2)
        - Summer: Mar, Apr, May (3, 4, 5)
        - Monsoon: Jun, Jul, Aug, Sep (6, 7, 8, 9)
        - Post-Monsoon: Oct, Nov (10, 11)
        """
        if month in (12, 1, 2):
            return "Winter"
        elif 3 <= month <= 5:
            return "Summer"
        elif 6 <= month <= 9:
            return "Monsoon"
        elif month in (10, 11):
            return "Post-Monsoon"
        else:
            return "Unknown"  # Fallback for invalid months
    
    # -----------------------------
    # COLLECT MONTHS FROM DATA
    # -----------------------------
    months = set()
    
    # Extract months from rows
    if rows:
        for r in rows:
            if not isinstance(r, dict):
                continue
                
            # Try to get date from row
            date_field = r.get("date")
            if date_field is None:
                continue
            
            try:
                # Handle different date formats
                if isinstance(date_field, str):
                    # Parse ISO format string
                    dt = datetime.fromisoformat(date_field.replace('Z', '+00:00'))
                    months.add(dt.month)
                elif isinstance(date_field, datetime):
                    months.add(date_field.month)
                elif hasattr(date_field, 'month'):
                    # Handle date objects
                    months.add(date_field.month)
            except (ValueError, AttributeError, TypeError):
                # Skip invalid date entries
                continue
    
    # Fallback to date_val if no months found in rows
    if not months and date_val is not None:
        try:
            if isinstance(date_val, str):
                dt = datetime.fromisoformat(date_val.replace('Z', '+00:00'))
                months.add(dt.month)
            elif isinstance(date_val, datetime):
                months.add(date_val.month)
            elif hasattr(date_val, 'month'):
                months.add(date_val.month)
        except (ValueError, AttributeError, TypeError):
            pass
    
    # Determine season(s)
    if len(months) == 0:
        season = None
    elif len(months) == 1:
        season = month_to_season(next(iter(months)))
    else:
        # Multiple months - check if all same season
        seasons = {month_to_season(m) for m in months}
        if len(seasons) == 1:
            season = next(iter(seasons))
        else:
            season = "Multiple Seasons"
    
    # -----------------------------
    # BLOCK PERIOD DEFINITIONS
    # -----------------------------
    BLOCK_PERIODS = {
        "Off-Peak (Night)":   (1, 24),    # 00:00–06:00
        "Morning Ramp":       (25, 36),   # 06:00–09:00
        "Morning Peak":       (37, 48),   # 09:00–12:00
        "Midday":             (49, 60),   # 12:00–15:00
        "Afternoon Peak":     (61, 72),   # 15:00–18:00
        "Evening Peak":       (73, 84),   # 18:00–21:00
        "Night Ramp-down":    (85, 96),   # 21:00–23:45
    }
    
    # -----------------------------
    # DETECT DEMAND COLUMN (CORRECTED)
    # -----------------------------
    demand_key = None
    
    if rows and len(rows) > 0 and isinstance(rows[0], dict):
        # Priority order for demand column detection
        demand_candidates = [
            "demand",
            "actual_demand", 
            "forecasted_demand",
            "predicted_demand",
            "value"
        ]
        
        first_row_keys = set(rows[0].keys())
        
        for candidate in demand_candidates:
            if candidate in first_row_keys:
                demand_key = candidate
                break
    
    # -----------------------------
    # BLOCK PERIOD STATISTICS (CORRECTED)
    # -----------------------------
    period_stats = {}
    
    # Only compute if we have rows, demand key, and block field
    if rows and demand_key and len(rows) > 0:
        # Check if block exists in first row
        if 'block' not in rows[0]:
            # No block data available
            return {
                "season": season,
                "period_stats": {},
                "demand_key": demand_key,
            }
        
        for period_name, (lo, hi) in BLOCK_PERIODS.items():
            values = []
            
            for r in rows:
                if not isinstance(r, dict):
                    continue
                
                try:
                    # Get block value with type checking
                    block_val = r.get("block")
                    if block_val is None:
                        continue
                    
                    # Convert to int safely
                    try:
                        b = int(block_val)
                    except (ValueError, TypeError):
                        continue
                    
                    # Check if block is in range
                    if not (lo <= b <= hi):
                        continue
                    
                    # Get demand value
                    demand_val = r.get(demand_key)
                    if demand_val is None:
                        continue
                    
                    # Convert to float safely
                    try:
                        v = float(demand_val)
                        values.append(v)
                    except (ValueError, TypeError):
                        continue
                        
                except Exception:
                    # Skip problematic rows
                    continue
            
            # Only add stats if we have values
            if values:
                period_stats[period_name] = {
                    "count": len(values),
                    "avg": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                }
    
    # -----------------------------
    # RETURN CONTEXT (CORRECTED)
    # -----------------------------
    return {
        "season": season,
        "period_stats": period_stats,
        "demand_key": demand_key,
    }


# -----------------------------
# HELPER: Get readable time from block number
# -----------------------------
def block_to_time_string(block: int) -> str:
    """
    Convert block number (1-96) to time string.
    Each block represents 15 minutes.
    
    Args:
        block: Block number (1-96)
    
    Returns:
        Time string in format "HH:MM"
    
    Examples:
        block_to_time_string(1)  -> "00:00"
        block_to_time_string(25) -> "06:00"
        block_to_time_string(96) -> "23:45"
    """
    if not isinstance(block, int) or block < 1 or block > 96:
        return "Invalid"
    
    # Block 1 = 00:00, Block 96 = 23:45
    # Each block = 15 minutes, starting from 00:00
    total_minutes = (block - 1) * 15
    hours = total_minutes // 60
    minutes = total_minutes % 60
    
    return f"{hours:02d}:{minutes:02d}"


# -----------------------------
# HELPER: Get block number from time
# -----------------------------
def time_to_block(hour: int, minute: int = 0) -> int:
    """
    Convert time to block number.
    
    Args:
        hour: Hour (0-23)
        minute: Minute (0-59)
    
    Returns:
        Block number (1-96)
    
    Examples:
        time_to_block(0, 0)   -> 1
        time_to_block(6, 0)   -> 25
        time_to_block(23, 45) -> 96
    """
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("Invalid time")
    
    total_minutes = hour * 60 + minute
    block = (total_minutes // 15) + 1
    
    return min(block, 96)  # Cap at 96


# -----------------------------
# HELPER: Get period name from block
# -----------------------------
def get_period_from_block(block: int) -> str:
    """
    Get period name from block number.
    
    Args:
        block: Block number (1-96)
    
    Returns:
        Period name as string
    """
    if not isinstance(block, int) or block < 1 or block > 96:
        return "Invalid"
    
    if 1 <= block <= 24:
        return "Off-Peak (Night)"
    elif 25 <= block <= 36:
        return "Morning Ramp"
    elif 37 <= block <= 48:
        return "Morning Peak"
    elif 49 <= block <= 60:
        return "Midday"
    elif 61 <= block <= 72:
        return "Afternoon Peak"
    elif 73 <= block <= 84:
        return "Evening Peak"
    elif 85 <= block <= 96:
        return "Night Ramp-down"
    else:
        return "Unknown"


# -------------------------
# INTENT IDENTIFIER (PLANNER)
# -------------------------
def intent_identifier_agent(state: GraphState) -> GraphState:
    """
    Plans execution path based on query analysis.
    Maps to 'Intent Identifier (Planner)' node in flowchart.
    """
    stage_print("PLANNER", "Planning execution path", model_name=get_agent_model_name("intent_identifier"))
    stage_print("PLANNER", f"Execution plan: intent={state.intent}", model_name=get_agent_model_name("intent_identifier"))
    return state


# -------------------------
# NL2SQL AGENT - IMPROVED with graph SQL storage
# -------------------------
def nl_to_sql_agent(state: GraphState) -> GraphState:
    """
    Converts natural language to SQL and executes query.
    Maps to 'NL to SQL AGENT' node in flowchart.
    """
    stage_print("NL2SQL", "Converting query to SQL", model_name=get_agent_model_name("nl_to_sql"))
    stage_print("NL2SQL", f"Calling DB tool with query: {state.user_query}", model_name=get_agent_model_name("nl_to_sql"))
    
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
    stage_print("NL2SQL", f"Success: {tool_result.get('row_count', 0)} rows retrieved", model_name=get_agent_model_name("nl_to_sql"))
    
    # IMPROVED: Store SQL for graph generation if needed
    sql = tool_result.get("sql", "")
    if state.need_graph and sql:
        stage_print("NL2SQL", "Storing SQL for graph generation", model_name=get_agent_model_name("nl_to_sql"))
        state.graph_data = {
            "sql": sql,
            "user_query": state.user_query
        }
    
    return state


# -------------------------
# DATA AND OBSERVATION AGENT - IMPROVED with graph support
# -------------------------
def data_observation_agent(state: GraphState) -> GraphState:
    """
    Handles historical data queries and observations.
    Maps to 'Data and observation AGENT' node in flowchart.
    """
    stage_print("DATA_OBSERVATION", "Processing data query", model_name=get_agent_model_name("data_observation"))

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

        stage_print("DATA_OBSERVATION", f"Dataset has {row_count} rows", model_name=get_agent_model_name("data_observation"))

        # For large datasets (>100 rows), generate aggregation query
        if row_count > 100:
            stage_print("DATA_OBSERVATION", "Large dataset detected - generating aggregation", model_name=get_agent_model_name("data_observation"))
            
            agg_sql = generate_aggregation_query(sql, state.user_query)
            stage_print("DATA_OBSERVATION", f"Aggregation SQL: {agg_sql}", model_name=get_agent_model_name("data_observation"))
            
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

        # IMPROVED: Prepare graph data if needed
        if state.need_graph and sql:
            stage_print("DATA_OBSERVATION", "Storing SQL for graph generation", model_name=get_agent_model_name("data_observation"))
            state.graph_data = {
                "sql": sql,
                "user_query": state.user_query
            }

        return state

    # If no data yet, call NL2SQL tool
    stage_print("DATA_OBSERVATION", "Calling DB tool", model_name=get_agent_model_name("data_observation"))
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

    # IMPROVED: Store graph data if needed
    if state.need_graph and sql:
        print("[DATA_OBSERVATION] Storing SQL for graph generation")
        state.graph_data = {
            "sql": sql,
            "user_query": state.user_query
        }

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


# -------------------------
# FORECASTING AGENT - IMPROVED with graph support
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
    if not state.need_model_run and not is_within_range(dt, FORECAST_DATA_START, FORECAST_DATA_END):
        state.data_ref = {
            "ok": False,
            "error_type": "out_of_range",
            "message": build_out_of_range_message("forecast"),
            "requested_date": requested_date
        }
        state.is_out_of_range = True
        return state

    # If the planner requested a model run, run the model
    if state.need_model_run:
        print(f"[FORECASTING] need_model_run=True → forcing model run for {requested_date}")

        run_result = model_run_tool.invoke(str(requested_date))

        if not run_result.get("ok"):
            state.data_ref = {
                "ok": False,
                "error_type": "model_run_failure",
                "message": run_result.get("error", "Model run failed")
            }
            return state

        # Fetch predictions from DB
        # To this (with proper schema prefix):
        if DB_TYPE == "postgresql":
            fetch_sql = (
                "SELECT datetime, block, predicted_demand AS forecasted_demand, model_id, generated_at "
                "FROM lf.t_predicted_demand_chatbot "
                f"WHERE prediction_date = '{requested_date}' "
                "ORDER BY datetime"
            )
        else:
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
        sql = state.data_ref.get("sql", "")
        
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
        
        # IMPROVED: Prepare graph if needed
        if state.need_graph and sql:
            print("[FORECASTING] Storing SQL for graph generation")
            state.graph_data = {
                "sql": sql,
                "user_query": state.user_query
            }
        
        return state

    # Otherwise, call NL2SQL tool
    print("[FORECASTING] Calling NL2SQL tool for forecast data")
    tool_result = nl_to_sql_db_tool.invoke(state.user_query)

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
    sql = tool_result.get("sql", "")
    
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
    
    # IMPROVED: Store graph data if needed
    if state.need_graph and sql:
        print("[FORECASTING] Storing SQL for graph generation")
        state.graph_data = {
            "sql": sql,
            "user_query": state.user_query
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
# UNIVERSAL SUMMARIZATION AGENT - IMPROVED with better graph handling
# -------------------------
def summarization_agent(state: GraphState) -> GraphState:
    """
    Universal gateway for ALL responses to frontend.
    Processes data from all agents and creates human-readable outputs.
    Maps to 'Summarization AGENT' node in flowchart.
    """
    stage_print("SUMMARIZATION", "Processing response for frontend delivery", llm=_get_llm())
    stage_print("SUMMARIZATION", f"Intent: {state.intent}", llm=_get_llm())
    stage_print("SUMMARIZATION", f"Has data_ref: {state.data_ref is not None}", llm=_get_llm())
    stage_print("SUMMARIZATION", f"Need graph: {state.need_graph}", llm=_get_llm())
    stage_print("SUMMARIZATION", f"Has graph_data: {state.graph_data is not None}", llm=_get_llm())
    
    # IMPROVED: Execute graph plotting if needed AND we have data
    if state.need_graph and state.graph_data and state.graph_data.get("sql"):
        stage_print("SUMMARIZATION", "✓ Executing graph plotting", llm=_get_llm())
        state = execute_graph_plotting(state)
    elif state.need_graph:
        stage_print("SUMMARIZATION", f"⚠️ Graph needed but missing data:", llm=_get_llm())
        stage_print("SUMMARIZATION", f"  - has graph_data: {state.graph_data is not None}", llm=_get_llm())
        if state.graph_data:
            stage_print("SUMMARIZATION", f"  - has SQL: {state.graph_data.get('sql') is not None}", llm=_get_llm())

    # Check if there's an error to handle
    if state.data_ref and not state.data_ref.get("ok", True):
        state.final_answer = format_error_message(state)
        return state
    
    # Route to appropriate formatter based on intent and data type
    if state.intent == "text":
        if not state.final_answer:
            state.final_answer = handle_text_response(state)
    
    elif state.intent == "data":
        state.final_answer = format_data_response(state)
    
    elif state.intent == "forecast":
        state.final_answer = format_forecast_response(state)
    
    elif state.intent == "decision":
        state.final_answer = format_decision_response(state)
    
    else:
        state.final_answer = state.final_answer or "I've processed your request."
    
    stage_print("SUMMARIZATION", f"Final answer ready: {len(state.final_answer)} chars", llm=_get_llm())
    if state.graph_data:
        stage_print("SUMMARIZATION", f"Graph status: {state.graph_data.get('ok', False)}", llm=_get_llm())
    
    return state


def execute_graph_plotting(state: GraphState) -> GraphState:
    """IMPROVED: Execute graph plotting with user_query context"""
    print("[GRAPH_EXEC] Starting graph generation")
    
    graph_metadata = state.graph_data
    sql = graph_metadata.get("sql")
    user_query = graph_metadata.get("user_query", state.user_query)
    
    if not sql:
        print("[GRAPH_EXEC] ERROR: No SQL in graph_data!")
        return state
    
    print(f"[GRAPH_EXEC] SQL: {sql[:100]}...")
    print(f"[GRAPH_EXEC] User query: {user_query[:100]}...")
    
    try:
        # IMPROVED: Call graph_plotting_tool with both SQL AND user_query
        plot_result = graph_plotting_tool.invoke({
            "sql": sql,
            "user_query": user_query
        })
        
        print(f"[GRAPH_EXEC] Result: ok={plot_result.get('ok')}")
        
        if plot_result.get("ok"):
            print("[GRAPH_EXEC] ✓✓✓ GRAPH SUCCESS!")
            state.graph_data = plot_result
            state.graph_data["plot_success"] = True
        else:
            print(f"[GRAPH_EXEC] Graph FAILED: {plot_result.get('error')}")
            state.graph_data["plot_success"] = False
            state.graph_data["plot_error"] = plot_result.get('error')
            
    except Exception as e:
        print(f"[GRAPH_EXEC] EXCEPTION: {e}")
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
    
    return "I'm here to help with load forecasting queries. What would you like to know?"


def format_data_response(state: GraphState) -> str:
    """Format data query responses using LLM for natural presentation"""
    data = state.data_ref or {}
    message_type = data.get("message_type", "unknown")
    
    if message_type == "no_data":
        return "I couldn't find any records matching your query in the database."
    
    row_count = data.get("row_count", 0)
    rows = data.get("rows", [])
    agg = data.get("aggregation", [])
    sql = data.get("sql", "")
    
    show_tech = should_show_technical_details(state.user_query)
    
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
        
        date_val = None
        if rows:
            date_val = rows[0].get("date")
        ctx = get_season_and_block_context(date_val, rows)

        response = _get_llm().invoke(messages).content
        
        if show_tech:
            response += f"\n\n**Technical Details:**\n```sql\n{sql}\n```\n"
            response += f"*Returned {row_count:,} records*"
        
        # IMPROVED: Add graph info if available
        if state.graph_data and state.graph_data.get("ok"):
            plot_result = state.graph_data
            response += (
                "\n\n📈 **Visualization Created**\n"
                f"Generated a {plot_result.get('plot_type', 'dynamic')} chart with {plot_result.get('data_points', 0)} data points."
            )
        elif state.graph_data and not state.graph_data.get("ok"):
            response += f"\n\n⚠️ Note: Visualization could not be created: {state.graph_data.get('error', 'Unknown error')}"
        
        return response
        
    except Exception as e:
        stage_print("SUMMARIZATION", f"LLM formatting error: {e}", llm=_get_llm())
        return format_data_basic(data, show_tech)


def format_data_basic(data: dict, show_tech: bool) -> str:
    """Fallback basic data formatting"""
    row_count = data.get("row_count", 0)
    agg = data.get("aggregation", [])
    
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
        
        if show_tech:
            sql = data.get("sql", "")
            response += f"\n\n**Technical Details:**\n```sql\n{sql}\n```"

        if data.get("generated_by_run") and not data.get("run_metrics"):
            response += "\n\n⚠️ No validation metrics available for forward-looking forecasts (actuals not yet observed)."

        # IMPROVED: Add graph info
        if state.graph_data and state.graph_data.get("ok"):
            plot_result = state.graph_data
            response += (
                f"\n\n📈 **Visualization Created**\n"
                f"Generated a {plot_result.get('plot_type', 'line')} chart with {plot_result.get('data_points', 0)} data points."
            )

        return response
        
    except Exception as e:
        stage_print("SUMMARIZATION", f"Forecast formatting error: {e}", llm=_get_llm())
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
    
    # IMPROVED: Add graph info
    if state.graph_data and state.graph_data.get("ok"):
        plot_result = state.graph_data
        response += (
            f"\n\n📈 **Visualization Created**\n"
            f"Generated a {plot_result.get('plot_type', 'chart')} to support this analysis."
        )
    
    return response


# -------------------------
# TEXT AGENT
# -------------------------
def text_agent(state: GraphState) -> GraphState:
    """
    Handles general text queries that don't require tools.
    """
    print("[TEXT] Handling general query")
    
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
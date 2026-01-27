#graph.py
from langgraph.graph import StateGraph, END
from state import GraphState
from agents import (
    query_analysis_agent,
    intent_identifier_agent,
    nl_to_sql_agent,
    data_observation_agent,
    forecasting_agent,
    decision_intelligence_agent,
    summarization_agent,
    text_agent,
)

# -------------------------
# ROUTING FUNCTIONS
# -------------------------

def route_from_query_analysis(state: GraphState):
    """
    Routes from Query Analysis to Intent Identifier (Planner).
    All queries go through the planner.
    """
    return "planner"


def route_from_planner(state: GraphState):
    """
    Routes from Intent Identifier (Planner) based on need_db_call flag.
    According to flowchart: "What 1) need for extraction 2) Need for DB Call"
    """
    # If database call is needed, go to NL2SQL first
    if state.need_db_call:
        return "nl2sql"
    
    # Otherwise route directly to appropriate agent
    if state.intent == "data":
        return "data_observation"
    elif state.intent == "forecast":
        return "forecasting"
    elif state.intent == "decision":
        return "decision_intelligence"
    else:
        return "text"


def route_after_nl2sql(state: GraphState):
    """
    Routes from NL2SQL agent to the appropriate specialized agent.
    """
    if state.intent == "forecast":
        return "forecasting"
    elif state.intent == "data":
        return "data_observation"
    elif state.intent == "decision":
        return "decision_intelligence"
    
    # Default to data observation if unclear
    return "data_observation"


def route_to_tools(state: GraphState):
    """
    Routes from specialized agents to tools (DB, Model Run, Graph Plotting, API).
    According to flowchart, agents can use multiple tools.
    """
    # This routing happens WITHIN the agents themselves
    # by calling the appropriate tools
    # After tool usage, route to summarization
    return "summarization"


def should_end(state: GraphState):
    """
    Determines if we should end after summarization.
    Graph plotting is handled within agents if needed.
    """
    return "end"


# -------------------------
# BUILD GRAPH
# -------------------------

def build_graph():
    """
    Builds the complete agentic workflow graph matching the flowchart.
    
    Flow:
    Start → Query Analysis → (Exit or Planner)
    Planner → (NL2SQL or Direct to Agents)
    NL2SQL → Specialized Agents
    Agents → Tools (DB, Model, API, Graph) → Summarization → End
    """
    
    graph = StateGraph(GraphState)

    # -------------------------
    # ADD NODES
    # -------------------------
    
    # Entry point
    graph.add_node("query_analysis", query_analysis_agent)
    
    # Planning
    graph.add_node("planner", intent_identifier_agent)
    
    # NL2SQL Agent (converts NL to SQL)
    graph.add_node("nl2sql", nl_to_sql_agent)
    
    # Specialized Agents (green nodes in flowchart)
    graph.add_node("data_observation", data_observation_agent)
    graph.add_node("forecasting", forecasting_agent)
    graph.add_node("decision_intelligence", decision_intelligence_agent)
    graph.add_node("text", text_agent)
    
    # Final summarization
    graph.add_node("summarization", summarization_agent)

    # -------------------------
    # SET ENTRY POINT
    # -------------------------
    graph.set_entry_point("query_analysis")

    # -------------------------
    # ADD EDGES
    # -------------------------
    
    # Query Analysis → Planner (all queries go through planner)
    graph.add_edge("query_analysis", "planner")

    # Planner → NL2SQL or Direct to Agents
    graph.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "nl2sql": "nl2sql",
            "data_observation": "data_observation",
            "forecasting": "forecasting",
            "decision_intelligence": "decision_intelligence",
            "text": "text",
        },
    )

    # NL2SQL → Specialized Agents
    graph.add_conditional_edges(
        "nl2sql",
        route_after_nl2sql,
        {
            "data_observation": "data_observation",
            "forecasting": "forecasting",
            "decision_intelligence": "decision_intelligence",
        },
    )

    # All specialized agents → Summarization
    graph.add_edge("data_observation", "summarization")
    graph.add_edge("forecasting", "summarization")
    graph.add_edge("decision_intelligence", "summarization")
    graph.add_edge("text", "summarization")

    # Summarization → End
    # Note: Graph plotting is handled within agents using graph_plotting_tool
    graph.add_edge("summarization", END)

    # Note: Tools (DB, Model Run, Graph Plotting, Live API) are called
    # WITHIN the agent functions, not as separate nodes in the graph

    return graph.compile()


# -------------------------
# HELPER: Execute Graph
# -------------------------

def run_graph(user_query: str):
    """
    Convenience function to execute the graph with a user query.
    """
    workflow = build_graph()
    
    initial_state = GraphState(user_query=user_query)
    
    result = workflow.invoke(initial_state)
    
    return result
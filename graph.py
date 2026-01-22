from langgraph.graph import StateGraph, END
from state import GraphState
from agents import (
    intent_agent,
    data_agent,
    text_agent,
    forecast_agent,
)

# -------------------------
# ROUTING LOGIC
# -------------------------
def route(state: GraphState):
    if state.intent == "data":
        return "data"
    if state.intent == "forecast":
        return "forecast"
    return "text"

# -------------------------
# BUILD GRAPH
# -------------------------
def build_graph():
    graph = StateGraph(GraphState)

    # -------------------------
    # Nodes
    # -------------------------
    graph.add_node("intent", intent_agent)
    graph.add_node("data", data_agent)
    graph.add_node("forecast", forecast_agent)
    graph.add_node("text", text_agent)

    # -------------------------
    # Entry point
    # -------------------------
    graph.set_entry_point("intent")

    # -------------------------
    # Conditional routing
    # -------------------------
    graph.add_conditional_edges(
        "intent",
        route,
        {
            "data": "data",
            "forecast": "forecast",
            "text": "text",
        },
    )

    # -------------------------
    # Termination
    # -------------------------
    graph.add_edge("data", END)
    graph.add_edge("forecast", END)
    graph.add_edge("text", END)

    return graph.compile()

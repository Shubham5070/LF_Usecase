from graph import build_graph
from state import GraphState


if __name__ == "__main__":
    graph = build_graph()

    print("\n🧠 LangGraph + Ollama LLaMA 3.2 Ready\n")

    while True:
        q = input("🧑 You > ")
        if q.lower() in ("exit", "quit"):
            break

        state = GraphState(user_query=q)
        result = graph.invoke(state)

        print("\n🤖 Assistant >")
        print(result["final_answer"])
        print("-" * 50)

# main.py (or similar)
def run_query(user_query: str) -> dict:
    """
    This calls your LangGraph / agents / tools
    and returns final result
    """
    result = graph.invoke({"query": user_query})
    return result

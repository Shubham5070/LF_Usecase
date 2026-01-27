#main.py
from graph import build_graph
from state import GraphState
from tools import cleanup_old_plots
import atexit


def run_query(user_query: str) -> dict:
    """
    Execute a single query through the enhanced agentic workflow.
    
    Args:
        user_query: The user's natural language question
        
    Returns:
        Complete state dictionary with results and plot paths
    """
    graph = build_graph()
    initial_state = GraphState(user_query=user_query)
    result = graph.invoke(initial_state)
    return result


def display_result(result: dict):
    """
    Display results in a user-friendly format.
    """
    print("\n🤖 Assistant >")
    
    # Display final answer
    final_answer = result.get("final_answer")
    if final_answer:
        print(final_answer)
    else:
        print("⚠️ No response generated. Please try rephrasing your question.")
    
    # Display graph info if available
    graph_data = result.get("graph_data")
    if graph_data and isinstance(graph_data, dict) and graph_data.get("ok"):
        print("\n" + "="*60)
        print("📊 VISUALIZATION DETAILS")
        print("="*60)
        print(f"Plot Type: {graph_data.get('plot_type', 'N/A')}")
        print(f"Data Points: {graph_data.get('data_points', 0):,}")
        print(f"File Location: {graph_data.get('filepath', 'N/A')}")
        print(f"Filename: {graph_data.get('filename', 'N/A')}")
        print("="*60)
    
    # Display warnings
    if result.get("is_out_of_range"):
        print("\n⚠️ Note: Requested data is outside available date range")
    
    print("\n" + "-" * 60 + "\n")


def main():
    """
    Main interactive loop with enhanced features.
    """
    # Clean up old plots on startup
    print("[STARTUP] Cleaning up old plot files...")
    cleanup_old_plots(days_to_keep=7)
    
    # Build graph once
    graph = build_graph()

    print("\n🧠 Enhanced Load Forecasting Agentic System")
    print("=" * 60)
    print("\n✨ NEW FEATURES:")
    print("  • Smart aggregation for large datasets (>100 rows)")
    print("  • High-quality graph generation with file saving")
    print("  • Plots saved to ./plots/ directory")
    print("  • Automatic cleanup of old temp tables")
    print("\n📊 CAPABILITIES:")
    print("  • Historical data queries with aggregations")
    print("  • Demand forecasting with statistics")
    print("  • Business intelligence & insights")
    print("  • Trend visualization (line, bar, scatter plots)")
    print("\n💡 EXAMPLES:")
    print("  • 'Show me a trend of actual demand for January 2025'")
    print("  • 'What's the forecast for 2025-03-15?'")
    print("  • 'Give me statistics on demand for last week'")
    print("  • 'Compare actual vs forecasted demand'")
    print("\nType 'exit', 'quit', or 'bye' to stop")
    print("Type 'cleanup' to remove old plot files\n")

    while True:
        try:
            q = input("🧑 You > ").strip()
            
            if not q:
                continue
            
            # Special commands
            if q.lower() in ("exit", "quit", "bye"):
                print("\n👋 Goodbye! Your plots are saved in ./plots/\n")
                break
            
            if q.lower() == "cleanup":
                cleanup_old_plots(days_to_keep=7)
                print("✅ Old plot files cleaned up\n")
                continue
            
            # Create initial state
            state = GraphState(user_query=q)
            
            # Execute workflow
            print()  # Blank line for readability
            result = graph.invoke(state)
            
            # Display result
            display_result(result)

        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!\n")
            break
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}\n")
            print("Please try again with a different query.\n")
            import traceback
            traceback.print_exc()


# Cleanup on exit
atexit.register(lambda: cleanup_old_plots(days_to_keep=7))


if __name__ == "__main__":
    main()
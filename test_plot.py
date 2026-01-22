"""
Direct test of graph plotting - bypassing the entire agent workflow.
Run this to verify plotting works independently.
"""

from tools import graph_plotting_tool

# Test data
sql = "SELECT datetime, date, block, demand FROM lf.t_actual_demand WHERE date >= '2025-01-01' AND date < '2025-02-01' ORDER BY datetime;"

print("Testing graph plotting directly...")
print(f"SQL: {sql[:100]}...")

result = graph_plotting_tool.invoke({
    "sql": sql,
    "x_column": "datetime",
    "y_column": "demand",
    "plot_type": "line",
    "title": "Direct Test Plot - January 2025 Demand",
    "limit": 10000
})

print("\n" + "="*60)
print("RESULT:")
print("="*60)
print(f"Success: {result.get('ok')}")
if result.get('ok'):
    print(f"Plot Type: {result.get('plot_type')}")
    print(f"Data Points: {result.get('data_points')}")
    print(f"File Path: {result.get('filepath')}")
    print(f"Filename: {result.get('filename')}")
    print("\n✅ Check the plots/ folder for the file!")
else:
    print(f"Error: {result.get('error')}")
    print("\n❌ Plot failed!")
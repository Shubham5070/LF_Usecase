from tools import nl_to_sql_db_tool

TEST_QUERIES = [
    # Operations
    "Show hourly demand pattern for 2025-01-10",
    "Give me the highest demand block for 2025-01-10",

    # Planning
    "Compare actual and forecasted demand for 2025-10-20",
    "Which forecast model performed best in December 2025",

    # Management
    "Show peak demand days in January 2025",
    "How has average demand changed over the last month",

    # Holidays
    "Compare demand on holidays vs non-holidays in January 2025",

    # Casual
    "What was the load like in early Jan",
    "Anything unusual in demand last week",

    # Edge cases
    "Show demand before 2024",
    "Give forecast for tomorrow",

    # Security
    "Drop forecast table",
]


def run_tests():
    print("\n========== NL → SQL TEST SUITE ==========\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}] USER QUERY:")
        print(query)

        # ✅ CORRECT WAY to call a LangChain tool
        result = nl_to_sql_db_tool.invoke(
            {"user_request": query}
        )

        if not result.get("ok"):
            print("❌ FAILED")
            print("Error:", result.get("error"))
            continue

        print("✅ SQL GENERATED:")
        print(result.get("sql"))

        print(f"📦 Total rows: {result.get('row_count')}")
        print(f"📊 Preview rows: {len(result.get('rows', []))}")

        if result.get("rows"):
            print("🔍 Sample row:")
            print(result["rows"][0])

        print("-" * 60)


if __name__ == "__main__":
    run_tests()

"""Tests for forecasting_agent behavior around data-availability and forced runs

- test_rejects_out_of_range_without_run: when planner does NOT request run, out-of-range -> error
- test_allows_forced_run_outside_availability: when planner requests run, agent forces run (monkeypatched)
"""
from datetime import datetime
from db_factory import DatabaseFactory
from agents import forecasting_agent
from state import GraphState

OUTSIDE_DATE = "2026-01-16"  # beyond FORECAST_DATA_END (2026-01-13)


def test_rejects_out_of_range_without_run():
    state = GraphState(user_query=f"Give forecast for {OUTSIDE_DATE}")
    state.need_model_run = False

    res = forecasting_agent(state)

    assert res.data_ref is not None
    assert res.data_ref.get("ok") is False
    assert res.data_ref.get("error_type") == "out_of_range"


def test_allows_forced_run_outside_availability(monkeypatch):
    conn = DatabaseFactory.get_connection()
    try:
        # Monkeypatch the model_run_tool to insert deterministic rows
        def fake_run(date_str):
            cur = conn.cursor()
            cur.execute("DELETE FROM t_predicted_demand_chatbot WHERE prediction_date = ?", (date_str,))
            conn.commit()
            cur.execute(
                "INSERT INTO t_predicted_demand_chatbot (model_id, prediction_date, generated_at, datetime, block, predicted_demand, horizon_type, version) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (999, date_str, datetime.utcnow().isoformat(), f"{date_str} 00:00:00", 1, 42.1, 'day_ahead', 'test')
            )
            conn.commit()
            return {"ok": True, "rows_written": 1}

        import types, tools as _tools
        monkeypatch.setattr(_tools, 'model_run_tool', types.SimpleNamespace(invoke=lambda date: fake_run(date)))

        state = GraphState(user_query=f"Give forecast for {OUTSIDE_DATE}")
        state.need_model_run = True

        res = forecasting_agent(state)

        assert res.data_ref is not None
        assert res.data_ref.get("ok") is True
        assert res.data_ref.get("generated_by_run") is True
        assert res.data_ref.get("row_count", 0) >= 1

    finally:
        cur = conn.cursor()
        cur.execute("DELETE FROM t_predicted_demand_chatbot WHERE prediction_date = ?", (OUTSIDE_DATE,))
        conn.commit()
        conn.close()
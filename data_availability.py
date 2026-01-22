# data_availability.py
from datetime import date, datetime
from typing import Optional, Tuple


# =========================================================
# MVP DATA AVAILABILITY CONFIG (HARD-CODED FOR NOW)
# =========================================================

ACTUAL_DATA_START = datetime(2025, 1, 1, 0, 0)
ACTUAL_DATA_END   = datetime(2026, 1, 13, 23, 45)

FORECAST_DATA_START = datetime(2025, 10, 14, 0, 0)
FORECAST_DATA_END   = datetime(2026, 1, 13, 23, 45)

HOLIDAY_START = date(2025, 1, 1)
HOLIDAY_END   = date(2026, 12, 25)

METRICS_START = date(2025, 10, 14)
METRICS_END   = date(2026, 1, 13)


# =========================================================
# HELPERS
# =========================================================

def is_within_range(
    value,
    start,
    end
) -> bool:
    return start <= value <= end


def build_out_of_range_message(intent: str) -> str:
    """
    User-facing message when requested date is outside availability.
    """
    return (
        "⚠️ Data not available for the requested period.\n\n"
        "Here is the data currently available:\n\n"
        f"- Actual demand & weather: {ACTUAL_DATA_START.date()} → {ACTUAL_DATA_END.date()}\n"
        f"- Forecasted demand & weather: {FORECAST_DATA_START.date()} → {FORECAST_DATA_END.date()}\n"
        f"- Holidays: {HOLIDAY_START} → {HOLIDAY_END}\n"
        f"- Metrics: {METRICS_START} → {METRICS_END}\n\n"
        "Please rephrase your query within these ranges."
    )

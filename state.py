# state.py
from pydantic import BaseModel
from typing import Optional, Any


class GraphState(BaseModel):
    user_query: str

    intent: Optional[str] = None

    data_ref: Optional[Any] = None
    final_answer: Optional[str] = None

    # ---- NEW ----
    is_out_of_range: bool = False

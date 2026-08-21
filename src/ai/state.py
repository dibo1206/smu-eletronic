"""수강신청 챗봇 LangGraph의 State 정의.

rag-system의 src/ai/state.py와 같은 역할 — 그래프 노드들이 주고받는
상태(State)를 한 곳에 모아둔다.
"""

from typing import Annotated, List, Optional, TypedDict

import pandas as pd
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    query_type: str
    sql: str
    pages: list
    category: str
    dataframe: Optional[pd.DataFrame]
    from_cache: bool
    fallback_used: bool
    insufficient: bool
    retry_count: int
    rewritten_query: Optional[str]
    primary_result: Optional[dict]

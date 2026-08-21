"""수강신청 챗봇의 LangGraph 조립.

rag-system의 src/ai/graph.py와 같은 역할 — state.py의 State와 nodes.py의
노드/라우팅 함수를 엣지로 연결해서 그래프를 완성한다. 실제 판단 로직은
전부 nodes.py에 있고, 여기서는 "어떤 노드를 어떤 순서로 잇는지"만 다룬다.
"""

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from ai.nodes import (
    answer_from_data,
    answer_from_document,
    classify_query,
    data_fallback,
    document_fallback,
    general_answer,
    rewrite_document_query,
    route_after_data,
    route_after_document,
    route_by_query_type,
)
from ai.state import State


def build_graph():
    builder = StateGraph(State)

    builder.add_node("classify_query", classify_query)
    builder.add_node("general", general_answer)
    builder.add_node("document", answer_from_document)
    builder.add_node("rewrite_document_query", rewrite_document_query)
    builder.add_node("data", answer_from_data)
    builder.add_node("document_fallback", document_fallback)
    builder.add_node("data_fallback", data_fallback)

    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges(
        "classify_query",
        route_by_query_type,
        {"general": "general", "document": "document", "data": "data"},
    )
    # 일반 대화는 검색 대상이 아니므로 바로 종료 (rag-system 베이스라인과 동일)
    builder.add_edge("general", END)

    # document: 부족하면 먼저 질문을 재작성해서 같은 도메인 안에서 재시도하고
    # (rag-system 베이스라인의 vector_search ↔ rewrite_query 루프),
    # 재시도가 다 떨어지면 그때 data로 폴백한다.
    builder.add_conditional_edges(
        "document",
        route_after_document,
        {"rewrite": "rewrite_document_query", "fallback": "data_fallback", "end": END},
    )
    builder.add_edge("rewrite_document_query", "document")

    # data: 부족하면 먼저 자기 자신을 다시 시도하고
    # (rag-system 베이스라인의 database_query 자기 자신 재시도 루프),
    # 재시도가 다 떨어지면 그때 document로 폴백한다.
    builder.add_conditional_edges(
        "data", route_after_data, {"retry": "data", "fallback": "document_fallback", "end": END}
    )

    builder.add_edge("document_fallback", END)
    builder.add_edge("data_fallback", END)

    return builder.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def ask(question: str) -> dict:
    """질문 하나를 그래프에 태워 답변/근거를 반환한다."""
    result = get_graph().invoke(
        {
            "messages": [HumanMessage(content=question)],
            "query_type": "",
            "sql": "",
            "pages": [],
            "category": "",
            "dataframe": None,
            "from_cache": False,
            "fallback_used": False,
            "insufficient": False,
            "retry_count": 0,
            "rewritten_query": None,
        }
    )
    return {
        "answer": result["messages"][-1].content,
        "query_type": result["query_type"],
        "sql": result.get("sql", ""),
        "pages": result.get("pages", []),
        "category": result.get("category", ""),
        "dataframe": result.get("dataframe"),
        "from_cache": result.get("from_cache", False),
        "fallback_used": result.get("fallback_used", False),
    }


# 그래프 인스턴스 (LangGraph Studio/CLI 등에서 바로 참조할 수 있도록 모듈 레벨에 노출)
graph = build_graph()

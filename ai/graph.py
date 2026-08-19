"""
수강신청 챗봇의 LangGraph 라우터.

rag-system의 09번(LangGraph 조건부 엣지) 노트북과 동일한 패턴:
1. classify_query 노드 — with_structured_output으로 질문을
   "document"(안내문 조회) vs "data"(DB 조회)로 분류
2. route_by_query_type — 분류 결과에 따라 다음 노드 결정
3. document 노드 → ai/rag.py, data 노드 → ai/text2sql.py 호출
"""

from enum import Enum
from pathlib import Path
from typing import Annotated, List, Literal, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ai.rag import get_rag_engine
from ai.text2sql import get_text2sql_engine

load_dotenv(Path(__file__).parent.parent / ".env")


class QueryType(str, Enum):
    DOCUMENT = "document"  # 수강신청 절차/유의사항/졸업요건 등 안내문 기반 질문
    DATA = "data"  # 특정 과목/시간표/경쟁률 등 DB 조회가 필요한 질문


class QueryClassification(BaseModel):
    query_type: QueryType = Field(description="조회 유형")
    reason: str = Field(description="분류 이유")


class State(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    query_type: str
    sql: str
    pages: list
    category: str


_llm = init_chat_model("gpt-5.4-mini")


def classify_query(state: State) -> dict:
    question = state["messages"][-1].content
    structured_llm = _llm.with_structured_output(QueryClassification)

    prompt = f"""다음 사용자 질문의 조회 유형을 분류하세요:

- document: 수강신청 절차, 정정기간, 유의사항, 졸업요건, 복수전공, 재수강 규정처럼
  학사 안내문에 담긴 "설명/규정"을 묻는 질문, 또는 교수님 본인의 소개 정보
  (세부전공, 연구실 위치, 학위, 개인 연락처)를 묻는 질문
  예: "졸업하려면 전공 몇 학점이야?", "재수강 규정이 어떻게 돼?",
      "이흥주 교수님 연구실이 어디야?", "조준희 교수님 세부전공이 뭐야?"

- data: 특정 과목/학년/시간표/경쟁률/강좌를 담당하는 교수님처럼 강좌 DB를
  조회해야 답할 수 있는 질문 (교수님이 어떤 과목을 몇 시에 가르치는지 등)
  예: "2학년 전공심화 과목 알려줘", "화요일에 열리는 강의는?",
      "정민철 교수님이 담당하는 과목이 뭐야?"

사용자 질문: {question}"""

    result = structured_llm.invoke(prompt)
    return {"query_type": result.query_type.value}


def route_by_query_type(state: State) -> Literal["document", "data"]:
    return state["query_type"]


def answer_from_document(state: State) -> dict:
    question = state["messages"][-1].content
    result = get_rag_engine().answer(question)
    return {
        "messages": [AIMessage(content=result["answer"])],
        "pages": result["pages"],
        "category": result.get("category") or "",
        "sql": "",
    }


def answer_from_data(state: State) -> dict:
    question = state["messages"][-1].content
    outcome = get_text2sql_engine().query(question)

    if outcome["error"]:
        answer = f"조회에 실패했어요: {outcome['error']}"
    else:
        summarize_prompt = f"""다음 SQL 실행 결과를 한국어로 한두 문장 요약하세요.
결과에 없는 내용은 추측하지 마세요.

질문: {question}
SQL: {outcome['sql']}
실행 결과: {outcome['result']}

요약:"""
        answer = _llm.invoke(summarize_prompt).content

    return {
        "messages": [AIMessage(content=answer)],
        "sql": outcome["sql"] or "",
        "pages": [],
        "category": "",
    }


def build_graph():
    builder = StateGraph(State)

    builder.add_node("classify_query", classify_query)
    builder.add_node("document", answer_from_document)
    builder.add_node("data", answer_from_data)

    builder.add_edge(START, "classify_query")
    builder.add_conditional_edges(
        "classify_query",
        route_by_query_type,
        {"document": "document", "data": "data"},
    )
    builder.add_edge("document", END)
    builder.add_edge("data", END)

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
        }
    )
    return {
        "answer": result["messages"][-1].content,
        "query_type": result["query_type"],
        "sql": result.get("sql", ""),
        "pages": result.get("pages", []),
        "category": result.get("category", ""),
    }
